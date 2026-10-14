import importlib, unittest
from types import SimpleNamespace
import torch
from ouro_integration.upstream import load_rome
from ouro_integration.moment_batching import physical_moment_batches

class PhysicalBatchTests(unittest.TestCase):
 def setUp(self):
  rome=load_rome();self.ls=importlib.import_module(rome.__package__+'.layer_stats')
 def test_conservation_no_long_sequence_truncation(self):
  items=[dict(input_ids=torch.arange(n)+1,attention_mask=torch.ones(n,dtype=torch.long),position_ids=torch.arange(n)) for n in [3,20,4,1]]
  original=self.ls.length_collation
  with physical_moment_batches(self.ls,8):
   batches=self.ls.length_collation(196608)(items)
   self.assertEqual(sum(int(b['attention_mask'].sum()) for b in batches),28)
   self.assertEqual([int(m.sum()) for b in batches for m in b['attention_mask']],[20,4,3,1])
   for b in batches:
    for ids,m in zip(b['input_ids'],b['attention_mask']):self.assertEqual(ids[m.bool()].tolist(),list(range(1,int(m.sum())+1)))
  self.assertIs(self.ls.length_collation,original)
 def test_fail_closed_and_exception_restore(self):
  original=self.ls.length_collation
  for bad in [0,-1,False,1.5]:
   with self.assertRaises(ValueError),physical_moment_batches(self.ls,bad):pass
  with self.assertRaisesRegex(RuntimeError,'intentional'):
   with physical_moment_batches(self.ls,8):
    with self.assertRaises(RuntimeError),physical_moment_batches(self.ls,4):pass
    raise RuntimeError('intentional')
  self.assertIs(self.ls.length_collation,original)
if __name__=='__main__':unittest.main()
