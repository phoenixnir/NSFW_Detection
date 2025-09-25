import numpy as np
import torch
import os.path
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import os


class AttnDataset(Dataset):
    def __init__(self, file_path, attn_path):
        self.attn_path = attn_path
        self.data = np.load(file_path, allow_pickle=True)

    def __len__(self):
        length = len(os.listdir(self.attn_path))
        return length

    def __getitem__(self, idx):
        path = os.path.join(self.attn_path, f'attn_{idx}.npy')
        attn = torch.tensor(np.load(path))
        
        attn = F.normalize(attn, p=2, dim=-1)
        
        prompt = self.data[idx]['prompt']
        label = self.data[idx]['label']
        cls = self.data[idx]['set']
        if label != 0:
            label = 1
        
        return attn, label, prompt, cls