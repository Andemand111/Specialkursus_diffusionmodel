#%%
import torch
import torch.nn as nn
#%%

""" Der er noget galt med tensorerne, de er ikke de rigtige størrelser.
    Det er omkring "x3" den går gal. Når den skal concatenates med x i decoderen.
"""

# Double convolution block

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8)
        self.conv_out = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, g):
        # Reshape for multihead attention and apply attention
        x_ = x.permute(2, 3, 0, 1).reshape(x.shape[2]*x.shape[3], x.shape[0], x.shape[1])
        g_ = g.permute(2, 3, 0, 1).reshape(g.shape[2]*g.shape[3], g.shape[0], g.shape[1])
        
        attn_output, _ = self.attention(query=x_, key=g_, value=g_)
        attn_output = attn_output.reshape(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2, 3, 0, 1)
        
        out = torch.cat((x, attn_output), dim=1)
        out = self.conv_out(out)
        
        return out

# Network architecture

class UNetWithAttention(nn.Module):
    def __init__(self):
        super(UNetWithAttention, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(4, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Attention Block
        self.attention = AttentionBlock(1024, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, 3, kernel_size=1)
        
    def forward(self, x, time_embedding):
        x = torch.cat((x, time_embedding), dim=1)  # Concatenate time embedding
        
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool(self.dropout(x1))
        x2 = self.enc2(x2)
        x3 = self.pool(self.dropout(x2))
        x3 = self.enc3(x3)
        x4 = self.pool(self.dropout(x3))
        x4 = self.enc4(x4)
        x5 = self.pool(self.dropout(x4))
        
        # Bottleneck
        x5 = self.bottleneck(x5)
        
        # Attention
        x5 = self.attention(x5, x5)
        
        # Decoder with skip connections
        x = self.upconv3(x5)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.outc(x)
        
        return x

model = UNetWithAttention()
print(model)

# %%
# make sure the model is trainable
for param in model.parameters():
    print(param.requires_grad)

# %%
# Define a random image tensor: (batch_size, channels, height, width)
batch_size = 8
img = torch.randn(batch_size, 3, 64, 64)

# Generate a random time embedding tensor: (batch_size, 1, 64, 64)
time_embedding = torch.randn(batch_size, 1, 64, 64)

# Instantiate the model and move to appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetWithAttention().to(device)

# Move the tensors to the same device
img, time_embedding = img.to(device), time_embedding.to(device)

# Forward pass
with torch.no_grad():
    output = model(img, time_embedding)

print(f"Input shape: {img.shape}")
print(f"Output shape: {output.shape}")



# %%