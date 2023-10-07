#set up 
import torch 
import torchvision
import matplotlib.pyplot as plt 
from torch import nn 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os 

device = "cpu"

#get data
image_path = "DermNet"
train_dir = "Dermnet/train"
test_dir = "Dermnet/test"

#preprocess data

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int= os.cpu_count()
):
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])
print(f"Manually created transforms: {manual_transforms}")

train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, 
    batch_size=32
)

print(f"{train_dataloader}, {test_dataloader}, {class_names}")

#Replicating ViT Architecture

class PatchEmbedding(nn.Module):

    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

       
        self.flatten = nn.Flatten(start_dim=2, 
                                  end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"
        
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        return x_flattened.permute(0, 2, 1)

patch_embedding_object = PatchEmbedding(patch_size=16)

#Transformer Layers

transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=512,
                                                       nhead=12,
                                                       dim_feedforward=2048,
                                                       dropout=0.1,
                                                       activation="relu",
                                                       batch_first=True,
                                                       norm_first=True)
transformer_encoder_layer
