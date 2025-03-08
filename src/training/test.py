import torch 
import json 
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.model import CLIP 
import matplotlib.pyplot as plt 
from src.data.dataset import FashionMNIST 
from src.models.text_encoder import tokenizer 
from torch.utils.data import DataLoader 

with open("src/data/parameters.json", "r") as file:
    parameters = json.load(file) 

test_set = FashionMNIST(train = False)
test_dataloader = DataLoader(
    test_set, shuffle = True
)
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

model.load_state_dict(torch.load('saved-models/CLIP.pt', map_location = device)) 

# Captions to compare images to
class_names =[
   "t-shirt/top",
    "trousers",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot"
]

text = torch.stack([tokenizer(x)[0] for x in class_names]).to(device)
mask = torch.stack([tokenizer(x)[1] for x in class_names])
mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

index = torch.randint(0, len(test_set), size = [1])

image = test_set[index.item()]['Image'][None,:]
plt.imshow(image[0].permute(1, 2, 0), cmap = "gray") 
plt.title(tokenizer(test_set[index.item()]['Caption'], 
                    encode = False, mask = test_set[index.item()]["Mask"][0])[0]) 
plt.show()

image = image.to(device) 
with torch.no_grad():
    image_features = model.image_encoder(image) 
    text_features = model.text_encoder(text, mask = mask)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{class_names[int(index)]:>16s}: {100 * value.item():.2f}%")