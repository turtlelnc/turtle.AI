import tempfile,unittest
from dataclasses import replace
import torch
from models.resonant import Config,ResonantLM
from models.streaming_memory import MemoryConfig,ResonantStream
from models.page_readout import dense_page_bias,preserve_recall_mass

class PageReadoutTests(unittest.TestCase):
    def test_single_page_bias_zero(self):
        q=torch.randn(1,8,4);k=torch.randn_like(q)
        chosen=torch.zeros(1,8,1,dtype=torch.long)
        out=dense_page_bias(q,k,chosen,torch.arange(8)//4,
                            torch.zeros(8,8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool),3)
        torch.testing.assert_close(out,torch.zeros_like(out))

    def test_gradient_causality_and_stream(self):
        torch.set_num_threads(1);torch.manual_seed(27)
        model=ResonantLM(Config(vocab_size=16,width=8,layers=3,block_size=3,max_tokens=32,
                               gist_local_only=True,page_score_weight=1.))
        memory=MemoryConfig(block_size=3,working_blocks=1,sink_tokens=1,gist_slots=1,cold_bytes=120,top_k=2)
        ids=torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1]])
        result=model(ids,ids,memory=memory,disk_backed=True)
        result['losses']['total'].backward()
        self.assertTrue(torch.isfinite(model.gist_embedding.grad).all())
        model.eval()
        for dtype in (torch.float32,torch.float16):
            current=replace(memory,working_dtype=dtype)
            with torch.no_grad():
                expected=model(ids,memory=current,disk_backed=True)['logits'][0]
                prefix=model(ids[:,:10],memory=current,disk_backed=True)['logits'][0]
            tolerance=5e-4 if dtype==torch.float16 else 4e-6
            torch.testing.assert_close(expected[:10],prefix,atol=tolerance,rtol=tolerance)
            with tempfile.TemporaryDirectory() as directory:
                stream=ResonantStream(model,current,page_directory=directory)
                actual=torch.stack([stream.push(int(x)) for x in ids[0]])
            torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)

    def test_preserves_local_probabilities_and_empty_gradients(self):
        old=torch.tensor([[2.,1.,0.,-1.],[1.,2.,3.,4.]],requires_grad=True)
        bias=torch.tensor([[0.,0.,3.,-3.],[0.,0.,0.,0.]])
        remote=torch.tensor([[False,False,True,True],[False,False,False,False]])
        new=preserve_recall_mass(old,old+bias,remote)
        before=old.softmax(-1);after=new.softmax(-1)
        torch.testing.assert_close(before[~remote],after[~remote])
        torch.testing.assert_close((before*remote).sum(-1),(after*remote).sum(-1))
        after[:,2].sum().backward();self.assertTrue(torch.isfinite(old.grad).all())
        extra=bias.clone().requires_grad_(True)
        corrected=preserve_recall_mass(old.detach(),old.detach()+extra,remote).softmax(-1)
        (corrected*~remote).sum().backward()
        torch.testing.assert_close(extra.grad,torch.zeros_like(extra),atol=2e-7,rtol=0)

    def test_detached_control_same_forward_different_gradient(self):
        torch.manual_seed(28)
        q=torch.randn(1,8,4,requires_grad=True);k=torch.randn(1,8,4,requires_grad=True)
        chosen=torch.tensor([[[0,1]]*8])
        owners=torch.arange(8)//4;local=torch.zeros(8,8,dtype=torch.bool);gist=torch.zeros(8,dtype=torch.bool)
        direct=dense_page_bias(q,k,chosen,owners,local,gist,3,False)
        detached=dense_page_bias(q,k,chosen,owners,local,gist,3,True)
        torch.testing.assert_close(direct,detached,atol=0,rtol=0)
        self.assertFalse(detached.requires_grad)
        direct[:,:,0].sum().backward()
        self.assertGreater(q.grad.abs().sum().item(),0)
        self.assertGreater(k.grad[:,3].abs().sum().item(),0)
        self.assertEqual(k.grad[:,:3].abs().sum().item(),0)

if __name__=='__main__':unittest.main()
