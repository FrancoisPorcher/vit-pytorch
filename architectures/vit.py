import torch 
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            # norm -> linear -> activation -> dropout -> linear -> dropout
            # we first norm with a layer norm
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            # we project in a higher dimension hidden_dim
            nn.GELU(),
            # we apply the GELU activation function
            nn.Dropout(dropout),
            # we apply dropout
            nn.Linear(hidden_dim, dim),
            # we project back to the original dimension dim
            nn.Dropout(dropout)
            # we apply dropout
        )

    def forward(self, x):
        return self.net(x)
    

    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # Calculate the total inner dimension based on the number of attention heads and the dimension per head
        
        # Determine if a final projection layer is needed based on the number of heads and dimension per head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # Store the number of attention heads
        self.scale = dim_head ** -0.5  # Scaling factor for the attention scores (inverse of sqrt(dim_head))

        self.norm = nn.LayerNorm(dim)  # Layer normalization to stabilize training and improve convergence

        self.attend = nn.Softmax(dim=-1)  # Softmax layer to compute attention weights (along the last dimension)
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization during training

        # Linear layer to project input tensor into queries, keys, and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Conditional projection layer after attention, to project back to the original dimension if required
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # Linear layer to project concatenated head outputs back to the original input dimension
            nn.Dropout(dropout)         # Dropout layer for regularization
        ) if project_out else nn.Identity()  # Use Identity (no change) if no projection is needed

    def forward(self, x):
        x = self.norm(x)  # Apply normalization to the input tensor

        # Apply the linear layer to get queries, keys, and values, then split into 3 separate tensors
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Chunk the tensor into 3 parts along the last dimension: (query, key, value)

        # Reshape each chunk tensor from (batch_size, num_patches, inner_dim) to (batch_size, num_heads, num_patches, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Calculate dot products between queries and keys, scale by the inverse square root of dimension
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # Shape: (batch_size, num_heads, num_patches, num_patches)

        # Apply softmax to get attention weights
        attn = self.attend(dots)  # Shape: (batch_size, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)

        # Multiply attention weights by values to get the output
        out = torch.matmul(attn, v)  # Shape: (batch_size, num_heads, num_patches, dim_head)

        # Rearrange the output tensor to (batch_size, num_patches, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')  # Combine heads dimension with the output dimension

        # Project the output back to the original input dimension if needed
        out = self.to_out(out)  # Shape: (batch_size, num_patches, dim)

        return out  # Return the final output tensor
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim_ratio, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        mlp_dim = mlp_dim_ratio * dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return self.norm(x)
    
def pair(t):
    """
    Converts a single value into a tuple of two values.
    If t is already a tuple, it is returned as is.
    
    Args:
        t: A single value or a tuple.
    
    Returns:
        A tuple where both elements are t if t is not a tuple.
    """
    return t if isinstance(t, tuple) else (t, t)
    
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, pool='cls', channels=3, dim_head=64, dropout=0.):
        """
        Initializes a Vision Transformer (ViT) model.
        
        Args:
            image_size (int or tuple): Size of the input image (height, width).
            patch_size (int or tuple): Size of each patch (height, width).
            num_classes (int): Number of output classes.
            dim (int): Dimension of the embedding space.
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the feedforward network.
            pool (str): Pooling strategy ('cls' or 'mean').
            channels (int): Number of input channels (e.g., 3 for RGB images).
            dim_head (int): Dimension of each attention head.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # Convert image size and patch size to tuples if they are single values
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Ensure that the image dimensions are divisible by the patch size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # Calculate the number of patches and the dimension of each patch
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width


        # Define the patch embedding layer
        self.to_patch_embedding = nn.Sequential(
            # Rearrange the input tensor to (batch_size, num_patches, patch_dim)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),  # Normalize each patch
            nn.Linear(patch_dim, dim),  # Project patches to embedding dimension
            nn.LayerNorm(dim)  # Normalize the embedding
        )

        # Ensure the pooling strategy is valid
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Define CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # Learnable class token

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Positional embeddings for patches and class token

        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

        # Define the transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim_ratio, dropout)

        # Pooling strategy ('cls' token or mean of patches)
        self.pool = pool
        # Identity layer (no change to the tensor)
        self.to_latent = nn.Identity()

        # Classification head
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """
        Forward pass through the Vision Transformer model.
        
        Args:
            img (Tensor): Input image tensor of shape (batch_size, channels, height, width).
        
        Returns:
            dict: A dictionary containing the class token, feature map, and classification result.
        """
        # Convert image to patch embeddings
        x = self.to_patch_embedding(img) # Shape: (batch_size, num_patches, dim)
        b, n, _ = x.shape  # Get batch size, number of patches, and embedding dimension
        
        # Repeat class token for each image in the batch
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        
        # Concatenate class token with patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings to the input
        x += self.pos_embedding[:, :(n + 1)]
        
        # Apply dropout for regularization
        x = self.dropout(x)

        # Pass through transformer encoder
        x = self.transformer(x) # Shape: (batch_size, num_patches + 1, dim)

        # Extract class token and feature map
        cls_token = x[:, 0]  # Extract class token
        feature_map = x[:, 1:]  # Remaining tokens are feature map

        # Apply pooling operation: 'cls' token or mean of patches
        pooled_output = cls_token if self.pool == 'cls' else feature_map.mean(dim=1)

        # Apply the identity transformation (no change to the tensor)
        pooled_output = self.to_latent(pooled_output)

        # Apply the classification head to the pooled output
        classification_result = self.mlp_head(pooled_output)

        # Return a dictionary with the required components
        return {
            'cls_token': cls_token,  # Class token
            'feature_map': feature_map,  # Feature map (patch embeddings)
            'classification_head_logits': classification_result  # Final classification result
        }