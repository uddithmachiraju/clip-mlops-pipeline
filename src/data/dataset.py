from datasets import load_dataset 
from src.models.text_encoder import tokenizer 
import torchvision.transforms as transform
from torch.utils.data import Dataset 

class FashionMNIST(Dataset):
    def __init__(self, train = True): 
        super().__init__()
        self.dataset = load_dataset('fashion_mnist') 
        self.transform = transform.ToTensor() 
        if train: 
            self.split = 'train'
        else:
            self.split = 'test' 

        self.captions = {
            0 : "T-Shirt/Top",
            1 : "Trousers",
            2 : "Pullover",
            3 : "Dress", 
            4 : "Coat",
            5 : "Sandal",
            6 : "Shirt",
            7 : "Sneaker",
            8 : "Bag",
            9 : "Ankle Boot" 
        }

    def __len__(self):
        return self.dataset.num_rows[self.split] 
    
    def __getitem__(self, index):
        image = self.dataset[self.split][index]['image']
        image = self.transform(image) 

        caption, mask = tokenizer(
            self.captions[self.dataset[self.split[index]['label']]]
        )

        mask = mask.repeat(len(mask), 1) 

        return {
            'Image' : image,
            'Caption' : caption, 
            'Mask' : mask
        }