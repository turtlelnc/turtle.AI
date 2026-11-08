import unittest,tempfile
import torch
from models.resonant import Config,ResonantLM
from models.streaming_memory import MemoryConfig,ResonantStream
from tools.probe_gist_relations import documents,encode,fact_labels

class GistProbeTests(unittest.TestCase):
    def test_role_swap_and_bag_control(self):
        torch.set_num_threads(1);torch.manual_seed(55)
        model=ResonantLM(Config(vocab_size=324,width=8,layers=3,block_size=8,max_tokens=50)).eval()
        memory=MemoryConfig(block_size=8)
        ids=documents(56,2,5,'mixed');other=ids.clone()
        other[:,:40]=other[:,:40].reshape(2,5,2,4).flip(2).flatten(1)
        a,y=encode(model,memory,ids,5,['raw_bag','oracle_value'])
        b,z=encode(model,memory,other,5,['raw_bag','oracle_value'])
        self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(a['raw_bag'],b['raw_bag']))
        self.assertTrue(torch.equal(a['oracle_value'],b['oracle_value']))
        historical,_=encode(model,memory,ids,5,['raw_bag'],True)
        expected=a['raw_bag'].reshape(2,5,-1)[:,:3].reshape(6,-1)
        self.assertTrue(torch.equal(historical['raw_bag'],expected))

    def test_features_equal_actual_cached_gist_kv(self):
        torch.set_num_threads(1);torch.manual_seed(57)
        model=ResonantLM(Config(vocab_size=324,width=8,layers=3,block_size=8,max_tokens=50,gist_local_only=True)).eval()
        memory=MemoryConfig(block_size=8,working_blocks=1,sink_tokens=1,gist_slots=2,cold_bytes=1024)
        ids=documents(58,1,5,'mixed')
        features,_=encode(model,memory,ids,5,['gist_k_2','gist_v_2','gist_kv_2','body_kv_2','body_int4_kv_2'])
        with tempfile.TemporaryDirectory() as directory:
            stream=ResonantStream(model,memory,page_directory=directory);stream.prefill(ids[0].tolist())
            page=next(p for p in stream.pools[2].cold if p.block_id==1)
            torch.testing.assert_close(page.gist.key,features['gist_k_2'][1],atol=3e-6,rtol=3e-6)
            torch.testing.assert_close(page.gist.value,features['gist_v_2'][1],atol=3e-6,rtol=3e-6)
            actual=torch.cat((page.keys.decode(),page.values.decode()),-1).flatten()
            torch.testing.assert_close(actual,features['body_int4_kv_2'][1],atol=4e-6,rtol=4e-6)
            for summary in stream.pools[2].gists:
                page=summary.position//9
                torch.testing.assert_close(summary.key,features['gist_k_2'][page],atol=3e-6,rtol=3e-6)
                torch.testing.assert_close(summary.value,features['gist_v_2'][page],atol=3e-6,rtol=3e-6)
                torch.testing.assert_close(torch.cat((summary.key,summary.value)),features['gist_kv_2'][page],atol=3e-6,rtol=3e-6)

if __name__=='__main__':unittest.main()
