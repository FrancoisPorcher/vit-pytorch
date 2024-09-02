# food101_dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

# Define class label mapping
class_label_mapping = {
    "apple_pie": 0, "baby_back_ribs": 1, "baklava": 2, "beef_carpaccio": 3,
    "beef_tartare": 4, "beet_salad": 5, "beignets": 6, "bibimbap": 7,
    "bread_pudding": 8, "breakfast_burrito": 9, "bruschetta": 10,
    "caesar_salad": 11, "cannoli": 12, "caprese_salad": 13, "carrot_cake": 14,
    "ceviche": 15, "cheesecake": 16, "cheese_plate": 17, "chicken_curry": 18,
    "chicken_quesadilla": 19, "chicken_wings": 20, "chocolate_cake": 21,
    "chocolate_mousse": 22, "churros": 23, "clam_chowder": 24,
    "club_sandwich": 25, "crab_cakes": 26, "creme_brulee": 27,
    "croque_madame": 28, "cup_cakes": 29, "deviled_eggs": 30, "donuts": 31,
    "dumplings": 32, "edamame": 33, "eggs_benedict": 34, "escargots": 35,
    "falafel": 36, "filet_mignon": 37, "fish_and_chips": 38, "foie_gras": 39,
    "french_fries": 40, "french_onion_soup": 41, "french_toast": 42,
    "fried_calamari": 43, "fried_rice": 44, "frozen_yogurt": 45,
    "garlic_bread": 46, "gnocchi": 47, "greek_salad": 48,
    "grilled_cheese_sandwich": 49, "grilled_salmon": 50, "guacamole": 51,
    "gyoza": 52, "hamburger": 53, "hot_and_sour_soup": 54, "hot_dog": 55,
    "huevos_rancheros": 56, "hummus": 57, "ice_cream": 58, "lasagna": 59,
    "lobster_bisque": 60, "lobster_roll_sandwich": 61, "macaroni_and_cheese": 62,
    "macarons": 63, "miso_soup": 64, "mussels": 65, "nachos": 66,
    "omelette": 67, "onion_rings": 68, "oysters": 69, "pad_thai": 70,
    "paella": 71, "pancakes": 72, "panna_cotta": 73, "peking_duck": 74,
    "pho": 75, "pizza": 76, "pork_chop": 77, "poutine": 78, "prime_rib": 79,
    "pulled_pork_sandwich": 80, "ramen": 81, "ravioli": 82, "red_velvet_cake": 83,
    "risotto": 84, "samosa": 85, "sashimi": 86, "scallops": 87,
    "seaweed_salad": 88, "shrimp_and_grits": 89, "spaghetti_bolognese": 90,
    "spaghetti_carbonara": 91, "spring_rolls": 92, "steak": 93,
    "strawberry_shortcake": 94, "sushi": 95, "tacos": 96, "takoyaki": 97,
    "tiramisu": 98, "tuna_tartare": 99, "waffles": 100
}

# Define the Food101Dataset class
class Food101Dataset(Dataset):
    """Custom Dataset for Food-101 images and labels."""
    def __init__(self, dataset, class_label_mapping, transform=None):
        self.dataset = dataset
        self.class_label_mapping = class_label_mapping
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB")  # Ensure image is in RGB format
        label = item['label']
        
        # Apply the transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Map the label to the correct index
        label_name = list(self.class_label_mapping.keys())[label]
        label_idx = self.class_label_mapping[label_name]
        
        return image, label_idx

def get_food101_dataloader(batch_size=32, num_workers=0):
    """
    Load the Food101 dataset and return the train and validation dataloaders.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        train_loader, val_loader (DataLoader, DataLoader): DataLoaders for training and validation sets.
    """

    # Load the dataset
    ds = load_dataset("ethz/food101")
    train_data = ds["train"]
    validation_data = ds["validation"]

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224, typical size for models like ResNet
        transforms.ToTensor(),          # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Create the custom Dataset
    train_dataset = Food101Dataset(train_data, class_label_mapping, transform)
    validation_dataset = Food101Dataset(validation_data, class_label_mapping, transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader