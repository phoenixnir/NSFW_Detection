import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import DataLoader
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random
import tqdm
from collections import defaultdict
from utils.dataset import AttnDataset
from utils.get_embedding import get_text_embedding, get_image_embedding
import utils.modify_clip

device = 'cuda'

# nsfw_words = ['sexual', 'nude', 'sex', '18+', 'naked', 'Nudity and sexual content', 'porn', 'dick', 'vagina',
#                     'naked person', 'explicit content', 'uncensored', 'fuck', 'nipples',
#                     'nipples', 'naked breasts', 'areola']

text_dataset = np.load('text_dataset.npy', allow_pickle=True)

grouped_data = defaultdict(list)

for index, item in enumerate(text_dataset):
    grouped_data[item['set']].append((index, item))

selected_indices = []
for group in grouped_data.values():
    half_size = len(group) // 2
    selected = random.sample(group, half_size)
    selected_indices.extend([index for index, _ in selected])

all_indices = set(range(len(text_dataset)))
unselected_indices = list(all_indices - set(selected_indices))

data = []
labels = []
prompts = []
classes = []

dataset = AttnDataset('text_dataset.npy','attn')
dataloader = DataLoader(dataset, shuffle=False)

for data_item, label, prompt, cls in tqdm.tqdm(dataloader):
    data.append(data_item)
    labels.append(label)
    

raw_data = torch.stack(data).squeeze()
raw_labels = torch.tensor(labels).squeeze()

data = raw_data[selected_indices]
labels = raw_labels[selected_indices]
    
res = []
coefs = []
for k in range(12):
    for i in tqdm.tqdm(range(12)):
        X = data[:, k, i, :].squeeze().numpy()
        y = np.array(labels)
        
        lda = LinearDiscriminantAnalysis(n_components=1)
        
        X_r = lda.fit(X, y).transform(X)
        res.append(X_r)
        coef = lda.coef_/np.linalg.norm(lda.coef_)
        coefs.append(coef)

np.save(f'coefs.npy', coefs)