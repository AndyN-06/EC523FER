import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom dataset class for AffectNet CSV
class AffectNetCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the row corresponding to idx
        row = self.data.iloc[idx]
        image_path = row['image_path']
        # We use the expression label for facial emotion recognition
        expression = int(row['expression'])
        
        # Open the image and apply transforms
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, expression

# Define image transformations (adjust as needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Using ImageNet statistics; adjust if necessary.
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Patch Embedding: Splits image into patches and projects them to an embedding dimension.
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.proj(x)            # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)            # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)       # [B, num_patches, embed_dim]
        return x

# Vision Transformer for Facial Emotion Recognition
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=8,
                 embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embedding (one for each patch + cls token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head: normalization followed by a linear layer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        # Create patch embeddings from the image
        x = self.patch_embed(x)
        # Prepend the cls token to the patch embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embeddings
        x = x + self.pos_embed
        # PyTorch's transformer expects input as [sequence_length, batch_size, embedding_dim]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        # Extract the representation of the cls token (first token)
        x = x[0]
        x = self.mlp_head(x)
        return x

# Training loop
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    return model

if __name__ == "__main__":
    # Device configuration: Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to your generated CSV file
    csv_file = "affectnet.csv"  # Update with the correct path if needed
    
    # Create the dataset and dataloader
    dataset = AffectNetCSV(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Initialize the Vision Transformer model.
    # Adjust num_classes as needed based on your dataset (here assumed to be 8).
    model = VisionTransformer(num_classes=8, depth=6, num_heads=8)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train the model
    trained_model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)
