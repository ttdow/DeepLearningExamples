import numpy as np
import torch

training_dataset = torch.from_numpy(np.load('./data/training_data.pkl', allow_pickle=True))
testing_dataset = torch.from_numpy(np.load('./data/testing_data.pkl', allow_pickle=True))

print(len(training_dataset) / 1024)