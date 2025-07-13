import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

class SiameseDataset(Dataset):
    def __init__(self, image_dir, transform, useTriplets=True):
        self.transform = transform
        self.useTriplets = useTriplets
        self.class_names, self.class_images = self._load_data(image_dir)
        self.triplets = self._generate_triplets()
        
    def __len__(self):
        return len(self.triplets)
    
    def _load_data(self, image_dir):
        
        # Each folder in image_dir is a different class
        class_names = sorted(os.listdir(image_dir)) 

        # Get all image paths in the class folder per class
        class_images = {}
        for class_name in class_names:
            
            if class_name == '.DS_Store':
                continue
            
            class_folder_image = os.path.join(image_dir, class_name)
            image_files = sorted(os.listdir(class_folder_image))
            
            if len(image_files) == 0:
                print(f"WARNING: EMPTY FOLDER CLASS '{class_folder_image}'")
                continue
            if '.DS_Store' in image_files:
                image_files.remove('.DS_Store')
                
            image_paths = [os.path.join(class_folder_image, img) for img in image_files]
            class_images[class_name] = image_paths
            
        return class_names, class_images
    
    def __getitem__(self, idx):
        anchor, pos, neg = self.triplets[idx]

        # Open images
        anchor = Image.open(anchor).convert('RGB')
        pos = Image.open(pos).convert('RGB')
        neg = Image.open(neg).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)
        
        # If using triplets, return anchor, pos, neg
        if self.useTriplets:
             return anchor, pos, neg
        else:
            # Use pairs, with a 50% chance of returning a positive or negative pair
            if random.random() < 0.5:
                return anchor, pos, 1
            else:
                return anchor, neg, 0
        
    def _generate_triplets(self):
        
        # self.class_names: [class_name]
        # self.class_images: {class_name: [image_paths]}
        triplets = [] # anchor, pos, neg
        
        for class_name in self.class_names:
            
            # Skip classes with fewer than 2 images
            if len(self.class_images[class_name]) < 2:
                continue  
            
            # Get the pos and neg samples
            positive_samples = self.class_images[class_name]
            negative_classes = [cls for cls in self.class_names if cls != class_name]
            
            # for each anchor, get them a pos and neg
            for anchor in positive_samples:
                pos = random.choice([p for p in positive_samples if p != anchor])
                neg_class = random.choice(negative_classes)
                neg = random.choice(self.class_images[neg_class])
                
                triplets.append((anchor, pos,  neg))

        return triplets
       
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def denormalize(tensor, mean, std):
    """Reverses the normalization for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)  # Reshape to match C×H×W
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean  # Reverse normalization
    tensor = torch.clamp(tensor, 0, 1)  # Ensure values are in [0,1]
    return tensor

def get_dataloaders(root_path, bsize, useTriplets=True):
    # Loading datasets
    root_path = root_path[:-1] if root_path.endswith('/') else root_path
    train_dataset = DataLoader(SiameseDataset(f"{root_path}/train/", transform, useTriplets=useTriplets), batch_size=bsize, shuffle=True)
    test_dataset = DataLoader(SiameseDataset(f"{root_path}/test/", transform, useTriplets=useTriplets), batch_size=bsize)

    return train_dataset, test_dataset