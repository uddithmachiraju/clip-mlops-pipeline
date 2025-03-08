import json 
import torch 
from src.models.model import CLIP 
from flask import Flask, request, jsonify
from src.models.text_encoder import tokenizer
from torchvision.transforms import transforms 

with open("src/data/parameters.json", "r") as file:
    parameters = json.load(file) 

# Class names for predictions
class_names = [
    "t-shirt/top", "trousers", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot"
]

app = Flask(__name__) 

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

# Load pretrained weights
model.load_state_dict(torch.load("saved-models/CLIP.pt", map_location = device)) 
model.eval() 

# Prepare text encodings for the class names
text = torch.stack([tokenizer(x)[0] for x in class_names]).to(device) 
mask = torch.stack([tokenizer(x)[1] for x in class_names]) 
mask = mask.repeat(1, len(mask[0])).reshape(len(mask), len(mask[0]), len(mask[0])).to(device) 

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((parameters['image_size'], parameters['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

app.route("/")
def home():
    return jsonify(
        {"message" : "CLIP model API is running! upload a image in '/predict' page to predictions"} 
    )

app.route("/predict", methods = ['POST'])
def predict():
    if "image" not in request.files:
        return jsonify(
            {"Error" : "No Image uploaded"}
        ), 400 
    
    image = request.files["image"] 
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.image_encoder(image) 
            text_features = model.text_encoder(text, mask = mask) 

        # Normalize features 
        image_features /= image_features.norm(dim = 1, keepdim = True)
        text_features /= text_features.norm(dim = 1, keepdim = True) 

        # Compute similarity scores 
        similarity = (100 * image_features @ text_features.T).softmax(dim = -1)
        values, indices = similarity[0].topk(5)

        # Format predictions
        predictions = [
            {"Class" : class_names[int(index)], 
             "Confidence": f"{100 * value.item():.2f}%"}
             for value, index in zip(values, indices)
        ]
        return jsonify(
            {"Predictions": predictions}
        )
    except Exception as e:
        return jsonify(
            {"Error": str(e)}
        ), 500  
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000) 