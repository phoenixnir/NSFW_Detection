import numpy as np
import torch
from PIL import Image
import os.path
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import random
import tqdm
from collections import defaultdict
from utils.dataset import AttnDataset
from utils.get_embedding import get_text_embedding, get_image_embedding
import utils.modify_clip

device = 'cuda'

text_dataset = np.load('test_dataset.npy', allow_pickle=True)

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

print(raw_data.shape, raw_labels.shape)

data = raw_data[unselected_indices]
labels = raw_labels[unselected_indices]

coefs = np.load(f'coefs.npy')
coefs = coefs.reshape(12,12,768)
    
test_data = raw_data
test_labels = raw_labels


test_res = []
for k in range(12):
    for i in range(12):
        X = test_data[:, k, i, :].squeeze().numpy()
        
        X_r = coefs[k,i].dot(X.T).T
        test_res.append(X_r)
        

test_res = np.array(test_res)
test_res = np.swapaxes(test_res, 0, 1).squeeze()

test_label_res = np.zeros(test_res.shape[0])

for i in range(test_res.shape[0]):
    if np.mean(test_res[i]) > -0.03:
        test_label_res[i] = 1
    else:
        test_label_res[i] = 0
        
success_rate = 0
tpr = 0
tnr = 0
p_num = 0
n_num = 0
false_set = {}
mma_num=0

for i in range(len(test_label_res)):
    if test_labels[i] == 1:
        p_num += 1
        if test_label_res[i] == 1:
            tpr += 1
            success_rate += 1
    else:
        n_num += 1
        if test_label_res[i] == 0:
            tnr += 1
            success_rate += 1

success_rate = success_rate / len(test_label_res)
tpr = tpr / p_num
tnr = tnr / n_num
print(f'{success_rate}, {tpr}, {1-tnr}')