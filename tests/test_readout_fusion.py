import unittest,tempfile
import torch
from models.resonant import Config,ResonantLM
from models.streaming_memory import MemoryConfig,ResonantStream
from models.readout_fusion import fused_weights

class FusionTests(unittest.TestCase):
    def test_empty_recall_and_mass(self):
        scores=torch.tensor([[2.,1.,0.],[1.,0.,-1.]])
        visible=torch.ones_like(scores,dtype=torch.bool)
        remote=torch.tensor([[False,False,False],[False,True,True]])
        gate=torch.tensor([[.4],[.4]],requires_grad=True)
        out=fused_weights(scores,visible,remote,gate)
        torch.testing.assert_close(out[0],scores[0].softmax(-1))
        torch.testing.assert_close(out.sum(-1),torch.ones(2))
        self.assertGreaterEqual(float(out[1,0].detach()),float(scores[1].softmax(-1)[0])*.5)
        out[:,1].sum().backward();self.assertTrue(torch.isfinite(gate.grad).all())

    def test_stream_parity_and_gate_gradient(self):
        torch.set_num_threads(1);torch.manual_seed(29)
        model=ResonantLM(Config(vocab_size=16,width=8,layers=3,block_size=3,
                               max_tokens=32,gist_local_only=True,readout_fusion=True))
        memory=MemoryConfig(block_size=3,working_blocks=1,sink_tokens=1,gist_slots=1,cold_bytes=120,top_k=1)
        ids=torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13]])
        out=model(ids,ids,memory=memory,disk_backed=True)
        out['losses']['total'].backward()
        self.assertGreater(model.blocks[-1].attention.read_gate.weight.grad.abs().sum().item(),0)
        model.eval()
        with torch.no_grad():expected=model(ids,memory=memory,disk_backed=True)['logits'][0]
        with tempfile.TemporaryDirectory() as directory:
            stream=ResonantStream(model,memory,page_directory=directory)
            actual=torch.stack([stream.push(int(x)) for x in ids[0]])
        torch.testing.assert_close(actual,expected,atol=4e-6,rtol=4e-6)

    def test_fp16_stream_with_general_gist_policy(self):
        torch.set_num_threads(1);torch.manual_seed(31)
        model=ResonantLM(Config(vocab_size=16,width=8,layers=2,block_size=3,
                               max_tokens=16,readout_fusion=True)).eval()
        memory=MemoryConfig(block_size=3,working_blocks=1,sink_tokens=1,gist_slots=1,
                            cold_bytes=120,top_k=1,working_dtype=torch.float16)
        ids=torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
        with torch.no_grad():expected=model(ids,memory=memory,disk_backed=True)['logits'][0]
        with tempfile.TemporaryDirectory() as directory:
            stream=ResonantStream(model,memory,page_directory=directory)
            actual=torch.stack([stream.push(int(x)) for x in ids[0]])
        torch.testing.assert_close(actual,expected,atol=5e-4,rtol=5e-4)

if __name__=='__main__':unittest.main()
