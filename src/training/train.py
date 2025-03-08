import sys 
import os
import torch 
import json 
import numpy as np 
from tqdm import tqdm 
# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.model import CLIP 
from src.data.dataset import FashionMNIST 
from torch.utils.data import DataLoader 

with open("src/data/parameters.json", "r") as file:
    parameters = json.load(file) 

train_set = FashionMNIST(train = True) 
train_dataloader = DataLoader(
    train_set, shuffle = True, batch_size = parameters['batch_size']) 
device = 'cpu' 

model = CLIP(
    parameters['Emb_dim'], 
    parameters['vit_hidden_dim'],
    parameters['image_size'],
    parameters['patch_size'],
    parameters['n_channels'],
    parameters['vit_layers'],
    parameters['vit_heads'],
    parameters['vocab_size'],
    parameters['text_dim'],
    parameters['max_sequence'],
    parameters['text_heads'],
    parameters['text_layers']
).to(device)

optimizer = torch.optim.Adam(params = model.parameters(), 
                             lr = parameters['lr'])

best_loss = np.inf 
for epoch in range(parameters['epochs']):
    for index, data in enumerate(tqdm(train_dataloader), 0):
        image, caption, mask = data['Image'].to(device), data['Caption'].to(device), data['Mask'].to(device) 
        loss = model(image, caption, mask) 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

    if epoch % 10 == 0:
        print(f"Epoch: {epoch + 1} / {parameters['epochs']}, Loss: {loss.item():.3f}") 

    if loss.item() <= best_loss:
        best_loss = loss.item() 
        save_path = "saved-models/CLIP.pt"
        # Ensure 'saved-models/' directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the model
        torch.save(model.state_dict(), save_path)
        print(f'Model Saved at {loss.item()}') 

