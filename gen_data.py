import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import json
from utils.get_embedding import get_image_embedding, get_text_embedding
import utils.modify_clip
import os

device = 'cuda'

def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    return model, processor

model, processor = load_model()
model.to(device)

text_dataset = []

with open('dataset.json', 'r', encoding='utf-8') as file:
    text_dataset = json.load(file)

class AttnHook():
    def __init__(self, model):
        self.attn_out = []
        self.model = model
        self.current_layer = 0
        
    def forward_hook(self, module, input, output):
        out = torch.einsum("bnhc, dhc -> bnhd", output[2], output[3].reshape(768, 12, 768//12))
        bias_term = self.model.text_model.encoder.layers[self.current_layer].self_attn.out_proj.bias
        out = out + bias_term[np.newaxis, np.newaxis] / out.shape[1]
        self.current_layer += 1

        self.attn_out.append(out)
        
    def clear(self):
        self.attn_out.clear()
        self.current_layer = 0
    
batch_size = 16
prompts = [item['prompt'] for item in text_dataset]

attnhook = AttnHook(model)

hook_handles = []
for j in range(12):
    hook_handle = model.text_model.encoder.layers[j].self_attn.register_forward_hook(attnhook.forward_hook)
    hook_handles.append(hook_handle)

for i in range(0, len(prompts), batch_size):

    attnhook.clear()
    batch = prompts[i:i+batch_size]

    text_embedding, ids = get_text_embedding(batch, model, processor)

    res_attn_out = torch.stack(attnhook.attn_out, dim=1)
    
    if not os.path.exists(f'attn'):
        os.makedirs(f'attn')
    
    for j in range(res_attn_out.shape[0]):
        np_data = res_attn_out[j, :, ids[j]].detach().cpu().numpy()
        
        np.save(f'attn/attn_{i+j}.npy', np_data)

for j in range(12):
    hook_handles[j].remove()