import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import json
from utils.get_embedding import get_image_embedding, get_text_embedding
import utils.modify_clip
import os
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    return model, processor

model, processor = load_model()
model.to(device)

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


parser = argparse.ArgumentParser(description="Parameters of model")
parser.add_argument('--file_path', type=str, default="test_prompts.txt")
parser.add_argument('--prompt', type=str, default=None)
args = parser.parse_args()

if args.prompt!=None:
    prompts = [args.prompt]
else:
    with open(args.file_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]

batch_size = 16

attnhook = AttnHook(model)

hook_handles = []
for j in range(12):
    hook_handle = model.text_model.encoder.layers[j].self_attn.register_forward_hook(attnhook.forward_hook)
    hook_handles.append(hook_handle)

if len(prompts) > 1:
    for i in range(0, len(prompts), batch_size):

        attnhook.clear()
        all_data = []
        batch = prompts[i:i+batch_size]

        text_embedding, ids = get_text_embedding(batch, model, processor)

        res_attn_out = torch.stack(attnhook.attn_out, dim=1)
        
        for j in range(res_attn_out.shape[0]):
            np_data = res_attn_out[j, :, ids[j]].detach().cpu().numpy()
            all_data.append(np_data)
        
        coefs = np.load(f'coefs.npy')
        coefs = coefs.reshape(12,12,768)
            
        test_data = np.stack(all_data)

        test_res = []
        for k in range(12):
            for i in range(12):
                X = test_data[:, k, i, :]
                
                X_r = coefs[k,i].dot(X.T).T
                test_res.append(X_r)
                

        test_res = np.array(test_res)
        test_res = np.swapaxes(test_res, 0, 1).squeeze()

        test_label_res = np.zeros(test_res.shape[0])

        for i in range(test_res.shape[0]):
            if np.mean(test_res[i]) > -0.01:
                test_label_res[i] = 1
            else:
                test_label_res[i] = 0
        
        print(test_label_res)
else:
    attnhook.clear()
    all_data = []

    text_embedding, ids = get_text_embedding(prompts, model, processor)

    res_attn_out = torch.stack(attnhook.attn_out, dim=1)
    
    for j in range(res_attn_out.shape[0]):
        np_data = res_attn_out[j, :, ids[j]].detach().cpu().numpy()
        all_data.append(np_data)
    
    coefs = np.load(f'coefs.npy')
    coefs = coefs.reshape(12,12,768)
        
    test_data = np.stack(all_data)

    test_res = []
    for k in range(12):
        for i in range(12):
            X = test_data[:, k, i, :]
            
            X_r = coefs[k,i].dot(X.T).T
            test_res.append(X_r)
            

    test_res = np.array(test_res)


    if np.mean(test_res) > -0.01:
        test_label_res = 1
    else:
        test_label_res = 0
    
    print(test_label_res)
    

for j in range(12):
    hook_handles[j].remove()