# main.py

import argparse
import json
from architectures.vit import ViT
from data.imagenet_dataloader import get_imagenet_loaders
from training.train import train_vit
from utils.config_utils import save_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on ImageNet")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, default=3072, help="MLP dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")

    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageNet dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    # Other parameters
    parser.add_argument("--config_output", type=str, default="config.json", help="Path to save the configuration")

    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize the model
    model = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        channels=args.channels,
        dropout=args.dropout
    )

    # Get data loaders
    train_loader, val_loader = get_imagenet_loaders(args)

    # Train the model
    train_vit(model, train_loader, val_loader, args)

    # Save the configuration
    save_config(vars(args), args.config_output)

if __name__ == "__main__":
    main()