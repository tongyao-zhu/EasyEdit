import importlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from ouro_integration.upstream import load_rome
from ouro_integration.moment_shards import contiguous_moment_shard,merge_second_moments

class ShardTests(unittest.TestCase):
 def setUp(self):
  r=load_rome();self.ls=importlib.import_module(r.__package__+'.layer_stats');self.rs=importlib.import_module('_ouro_easyedit_upstream.util.runningstats')
 def test_exact_coverage_and_original_groups(self):
  data=torch.arange(400,dtype=torch.float32).reshape(200,2);order=list(self.rs.FixedRandomSubsetSampler(data,seed=1,end=200));seen=[];states=[]
  for start,end in [(0,100),(100,200)]:
   stat=self.rs.SecondMoment();group=[]
   with tempfile.TemporaryDirectory() as d,contiguous_moment_shard(self.ls,order,start,end):
    for x in self.ls.tally(stat,list(data),cache=Path(d)/'part.npz',sample_size=200,random_sample=1,batch_size=100):
     group.extend((x[:,0]/2).int().tolist());stat.add(x)
   self.assertEqual(group,order[start:end]);seen.extend(group);states.append(stat)
  self.assertEqual(seen,order);merged=merge_second_moments(states)
  torch.testing.assert_close(merged.moment(),data.T@data/len(data),rtol=1e-6,atol=1e-4);self.assertEqual(merged.count,200)
 def test_invalid_boundaries_and_restoration(self):
  old=self.ls.tally
  with self.assertRaises(ValueError),contiguous_moment_shard(self.ls,list(range(200)),1,100):pass
  with self.assertRaisesRegex(RuntimeError,'intentional'):
   with contiguous_moment_shard(self.ls,list(range(200)),0,100):raise RuntimeError('intentional')
  self.assertIs(self.ls.tally,old)
if __name__=='__main__':unittest.main()
