import numpy as np
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset): 
  def __init__(self, ages, weights, np_path):
    traces = np.load(np_path, "r+")
    print(traces.shape)
    print(ages.shape)
    print(weights.shape)
    
    self.traces = torch.from_numpy(traces)
    self.ages = torch.from_numpy(ages)
    self.weights = torch.from_numpy(weights)
    self.n_samples = traces.shape[0]
    
  def __len__(self): 
    return self.n_samples

  def __getitem__(self, idx): 
    trace = self.traces[idx]
    age = self.ages[idx]
    weight = self.weights[idx]    
    
    assert trace.shape == (5000, 8)

    return trace, age, weight