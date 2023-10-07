#set up 
import torch 
import torchvision
import matplotlib.pyplot as plt 
from torch import nn 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os 
from torchinfo import summary 


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


random_image_tensor = torch.randn(32, 3, 224, 224)

patch_embedding_object = PatchEmbedding(patch_size=16)
patch_embedding_obj_output = patch_embedding_object(random_image_tensor)

#Transformer Layers
transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                       nhead=12,
                                                       dim_feedforward=3072,
                                                       dropout=0.1,
                                                       activation="gelu",
                                                       batch_first=True,
                                                       norm_first=True)
transformer_encoder_layer

summary(model=transformer_encoder_layer, input_size=patch_embedding_obj_output.shape)

transformer_encoder = nn.TransformerEncoder(
    encoder_layer=transformer_encoder_layer,
    num_layers=12)

class VisionTransformer(nn.Module): 
  def __init__(self,
               img_size=224, 
               num_channels=3,
               patch_size=16,
               embedding_dim=768, 
               dropout=0.1, 
               mlp_size=3072, 
               num_transformer_layers=12, 
               num_heads=12,
               num_classes=1000): 
    super().__init__()

    assert img_size % patch_size == 0, "Image size must be divisble by patch size."

    self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

    self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                    requires_grad=True)

    num_patches = (img_size * img_size) // patch_size**2 # N = HW/P^2
    self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))
    self.embedding_dropout = nn.Dropout(p=dropout)

    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                              nhead=num_heads,
                                                                                              dim_feedforward=mlp_size,
                                                                                              activation="gelu",
                                                                                              batch_first=True,
                                                                                              norm_first=True),
                                                                                              num_layers=num_transformer_layers) 

    self.mlp_head = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.patch_embedding(x)
    class_token = self.class_token.expand(batch_size, -1, -1)
    x = torch.cat((class_token, x), dim=1)
    x = self.positional_embedding + x
    x = self.embedding_dropout(x)
    x = self.transformer_encoder(x)
    x = self.mlp_head(x[:, 0])
    return x

demo_img = torch.randn(1, 3, 224, 224).to(device)

vit = VisionTransformer(num_classes=len(class_names)).to(device)
vit(demo_img)

summary(model=VisionTransformer(num_classes=3),
        input_size=demo_img.shape)