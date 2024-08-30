# vit-pytorch

This is a PyTorch implementation of the Vision Transformer (ViT) model in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al.

# How to use?

Create a new conda environment called `vit` from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

# Vision Transformer (ViT) Training on ImageNet

This repository contains code for training a Vision Transformer (ViT) model on the ImageNet dataset. It provides a flexible and modular implementation that allows for easy experimentation with different hyperparameters and model configurations.

## Repository Structure

```
vit-pytorch/
├── architectures/
│   └── vit.py              # ViT model implementation
├── data/
│   └── imagenet_dataloader.py  # DataLoader for ImageNet
├── training/
│   └── train.py            # Trainer class and training logic
├── utils/
│   └── config_utils.py     # Utility functions for saving configs
├── main.py                 # Main script to run training
└── README.md               # This file
```

## Usage

To train the ViT model, use the following command:

```bash
python main.py --data_dir /path/to/imagenet/ --batch_size 64 --num_epochs 100 --learning_rate 3e-4 --weight_decay 0.1
```

This command will start training with the specified parameters. You can adjust the hyperparameters as needed.

## Example Configuration

Here's an example configuration you might use:

```bash
python main.py \
    --data_dir /path/to/imagenet/ \
    --image_size 224 \
    --patch_size 16 \
    --num_classes 1000 \
    --dim 768 \
    --depth 12 \
    --heads 12 \
    --mlp_dim 3072 \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --dropout 0.1
```

## Features

- Modular implementation of Vision Transformer
- Flexible training pipeline with customizable hyperparameters
- Automatic logging of training and validation metrics
- Visualization of training progress (loss and accuracy curves)
- Saves the best model based on validation accuracy
- Saves configuration for reproducibility

## Requirements

- PyTorch
- torchvision
- tqdm
- matplotlib
- numpy

## Notes

- Make sure you have enough computational resources, as training a ViT on ImageNet is computationally intensive.
- The code automatically uses GPU if available, otherwise falls back to CPU.
- Training progress is displayed in real-time, and a plot of training/validation loss and accuracy is saved at the end.
- The best model (based on validation accuracy) is saved as 'best_vit_model.pth'.
- The configuration used for training is saved as 'config.json' for future reference.

## Contributing

Feel free to open issues or pull requests if you have suggestions for improvements or encounter any problems.

## License

This project is open-source and available under the MIT License.