import argparse
from src.models.vit import ViT
from src.training.train import Trainer
from src.data.data_loader import get_cifar100_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for ViT")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=24, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension of each attention head")
    parser.add_argument("--mlp_dim", type=int, default=3072, help="Dimension of the MLP layer")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of output classes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cfg = {
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'dim_head': args.dim_head,
        'mlp_dim': args.mlp_dim,
        'num_classes': args.num_classes,
    }

    model = ViT(**cfg)

    train_loader, val_loader, test_loader = get_cifar100_dataloaders(batch_size=args.batch_size)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir
    )

    trainer.train()

    trainer.test()

    trainer.save_model()