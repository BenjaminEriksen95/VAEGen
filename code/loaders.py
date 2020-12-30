import torch
import numpy as np
from functools import reduce

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

def stratified_sampler(labels,classes):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

def get_target_indexes(dset, classes, n_examples):
  """Get indexes for n_examples of class in classes from dataset

  The returned indexes may be used for subsetting a dataset, e.g.
  if we just want classes 0, 1 and 2 and 10 examples of each.
  """
  idxs = []
  ts = []
  n_samples = n_examples * len(classes)

  for j, x in enumerate(dset.targets):
    if (ts.count(x.numpy()) < n_examples) and (x.numpy() in classes):
      ts.append(x.numpy())
      idxs.append(j)

    if len(idxs) == n_samples:
      break

  return idxs

# The loaders perform the actual work
def create_loader(dset, batch_size, classes=[0,1,2,3,4,5,6,7,8,9]):
    return DataLoader(dset, batch_size=batch_size,
                        sampler=stratified_sampler(dset.targets,classes))

# Make subsets
def create_subset(dset, batch_size, classes=[0,1,2,3,4,5,6,7,8,9], size=100):
    idx_subset = get_target_indexes(dset, classes, size)
    dset_subset = Subset(dset, idx_subset)

    return DataLoader(dset_subset, batch_size=batch_size)
