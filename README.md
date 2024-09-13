# vit-pytorch

This is a PyTorch implementation of the Vision Transformer (ViT) model in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al.

Additionally, it includes an implementation of a Masked Autoencoder (MAE) for advanced model training techniques.

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

## Creating a Vision Transformer (ViT)


To create a Vision Transformer with the appropriate parameters, follow the steps below. The ViT model architecture is defined in `architectures/vit.py.`

```python
from architectures.vit import ViT

vit = ViT(
    image_size=224,        # Size of the input images (height and width)
    patch_size=16,         # Size of each image patch
    num_classes=101,       # Number of output classes (e.g., 101 for Food101)
    dim=768,               # Embedding dimension
    depth=12,              # Number of transformer blocks
    heads=12,              # Number of attention heads
    mlp_dim_ratio=4,       # Ratio to determine MLP hidden dimension (dim * mlp_dim_ratio)
    dropout=0.1,           # Dropout rate
    emb_dropout=0.1        # Dropout rate for the embedding layer
)
```

Parameter Explanation

	•	image_size: The height and width of the input images. Typically set to 224 for standard image sizes.
	•	patch_size: The size of each image patch that the transformer will process. Commonly 16.
	•	num_classes: The number of classes for classification tasks. For Food101, this is 101.
	•	dim: The dimensionality of the token embeddings.
	•	depth: The number of transformer encoder layers.
	•	heads: The number of attention heads in each transformer block.
	•	mlp_dim_ratio: Determines the size of the hidden layer in the MLP relative to dim.
	•	dropout: Dropout rate applied after each attention and MLP layer.
	•	emb_dropout: Dropout rate applied to the input embeddings.

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





# Masked Autoencoder


The Masked Autoencoder (MAE) is integrated into this repository to facilitate self-supervised learning, enabling the model to learn representations by reconstructing masked portions of the input data.

## Implementation

Below is an example of how to create and use the MAE with the ViT encoder:

```python
import torch
from architectures.vit import ViT
from architectures.mae import MAE  # Ensure MAE is implemented in architectures/mae.py

# Initialize the Vision Transformer (ViT) model
vit = ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,      # Typically 1000 for ImageNet; adjust as needed
    dim=1024,
    depth=2,
    heads=8,
    mlp_dim_ratio=4,
    dropout=0.1,
    emb_dropout=0.1
)

# Create a sample input image tensor
sample_image = torch.randn(2, 3, 224, 224)  # Batch size of 2

# Initialize the Masked Autoencoder (MAE) with the ViT encoder
mae = MAE(
    encoder=vit,
    masking_ratio=0.75,      # Percentage of patches to mask
    decoder_dim=512,
    decoder_depth=6,
    decoder_heads=8,
    decoder_dim_head=64
)

# Perform a forward pass through the MAE without tracking gradients
with torch.no_grad():
    output = mae(sample_image)

print(output.shape)  # Output shape will depend on MAE implementation
```
### Parameters Explanation

	•	encoder: The Vision Transformer model used as the encoder.
	•	masking_ratio: The proportion of input patches to mask during training (e.g., 0.75 for 75% masking).
	•	decoder_dim: Embedding dimension for the decoder.
	•	decoder_depth: Number of transformer blocks in the decoder.
	•	decoder_heads: Number of attention heads in the decoder’s transformer blocks.
	•	decoder_dim_head: Dimension of each attention head in the decoder.

Ensure that the MAE class is properly implemented in architectures/mae.py or the appropriate module.

## Notes

- **Computational Resources:** Training a Vision Transformer, especially on large datasets like ImageNet, is computationally intensive. Ensure you have access to a GPU with sufficient memory to facilitate efficient training.
- **Hardware Acceleration:** The code automatically utilizes a GPU if available. If a GPU is not detected, it will default to CPU execution, which may significantly slow down training.
- **Training Visualization:** Training and validation loss, as well as accuracy metrics, are plotted and saved at the end of the training process for performance analysis.
- **Model Checkpointing:** The best-performing model based on validation accuracy is saved as best_vit_model.pth. Intermediate checkpoints can also be configured in the checkpoints directory.
- **Configuration Management:** All training configurations are saved in configs/config.json, ensuring that training setups can be replicated or modified easily.

## Contributing

Contributions are welcome! Whether you have suggestions for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request. Please ensure that your contributions adhere to the repository’s coding standards and include relevant tests.

## License

This project is open-source and available under the MIT License.

Feel free to reach out if you have any questions or need further assistance with the vit-pytorch project!