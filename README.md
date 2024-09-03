# vit-pytorch

This is a PyTorch implementation of the Vision Transformer (ViT) model in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al.

# How to use?

Create a new conda environment with `Pytorch`, `torchvision`, `tqdm`, and `einops` installed.



## Repository Structure

```bash
├── architectures
│   └── vit.py
├── best_vit_model.pth
├── checkpoints
├── configs
│   └── config.json
├── dataloader
│   └── food101_dataloader.py
├── dawn_cluster_scripts
│   └── train_vit.sh
├── logs
├── main.py
├── README.md
├── training
│   └── train.py
└── utils
    └── utils.py
```

## Training the ViT on Food101 dataset (Image Classification)

The dataset we are using is the Food101 dataset. It is a dataset of 101 food categories. For more information on the dataset, please refer to the following link:

https://huggingface.co/datasets/ethz/food101

## Usage

To train the ViT model, use the following command:

```bash
python main.py \
    --image_size 224 \
    --patch_size 16 \
    --num_classes 101 \
    --dim 768 \
    --depth 12 \
    --heads 12 \
    --mlp_dim_ratio 4 \
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