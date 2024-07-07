# vit-pytorch

This is a PyTorch implementation of the Vision Transformer (ViT) model in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al.

# How to use?

To launch training, you can run the `main.py` script. You can specify the following arguments:

```bash
python src/main.py \
    --image_size 32 \
    --patch_size 4 \
    --dim 768 \
    --depth 12 \
    --heads 8 \
    --dim_head 64 \
    --mlp_dim 3072 \
    --num_classes 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --checkpoint_dir ./checkpoints
```