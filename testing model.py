import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_channels // heads
        self.scale = self.head_dim ** -0.5  # Scaling factor
        
        # Linear transformations for Q, K, and V
        self.to_queries = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.to_keys = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.to_values = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output convolutional layer for merging the heads
        self.merge_heads = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Linearly transform the input to Q, K, and V
        queries = self.to_queries(x).view(B, self.heads, self.head_dim, H, W)
        keys = self.to_keys(x).view(B, self.heads, self.head_dim, H, W)
        values = self.to_values(x).view(B, self.heads, self.head_dim, H, W)
        
        # Permute dimensions for compatibility with batch matrix multiplication
        queries = queries.permute(0, 1, 3, 4, 2).contiguous().view(B * self.heads, H * W, self.head_dim)
        keys = keys.permute(0, 1, 3, 4, 2).contiguous().view(B * self.heads, H * W, self.head_dim)
        values = values.permute(0, 1, 3, 4, 2).contiguous().view(B * self.heads, H * W, self.head_dim)
        
        # Compute scaled dot-product attention
        attention = torch.matmul(queries, keys.permute(0, 2, 1)) * self.scale
        attention = torch.nn.functional.softmax(attention, dim=-1)
        out = torch.matmul(attention, values).view(B, self.heads, H, W, self.head_dim)
        
        # Merge the heads
        out = out.permute(0, 4, 1, 2, 3).contiguous().view(B, C, H, W)
        out = self.merge_heads(out)
        
        return out

class DeepConvSelfAttentionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, heads, num_layers):
        super(DeepConvSelfAttentionNetwork, self).__init__()
        
        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Stack of convolutional layers and self-attention blocks
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                SelfAttention(64, heads)
            )
            for _ in range(num_layers)
        ])
        
        # Output convolutional layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = torch.relu(self.initial_conv(x))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.out_conv(x)
        return x

# Example usage
n = 3  # Adjust this to the number of output channels you need
num_layers = 4  # Adjust the number of layers as desired
model = DeepConvSelfAttentionNetwork(n + 1, n, heads=4, num_layers=num_layers)

# Test with random input
input_image = torch.randn(1, n + 1, 28,28)  # Batch size 1, (n + 1) channels, 64x64 image
output_image = model(input_image)

print(output_image.size())  # Should be [1, n, 64, 64]