#set up 
import torch 
import torchvision
import matplotlib.pyplot as plt 
from torch import nn 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os 
from torchinfo import summary 
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

#get data
image_path = "DermNet"
train_dir = f"{image_path}/train"
test_dir = f"{image_path}/test"

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

train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, 
    batch_size=32
)

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

vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 
pretrained_vit = torchvision.models.vit_b_16(weights=vit_weights)

for param in pretrained_vit.parameters():
  param.requires_grad = False

embedding_dim = 768 

pretrained_vit.heads = nn.Sequential(
    nn.LayerNorm(normalized_shape=embedding_dim),
    nn.Linear(in_features=embedding_dim, 
              out_features=len(class_names))
)

summary(model=pretrained_vit, 
        input_size=(1, 3, 224, 224), 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

def train_test_step(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
   
    criterion = loss_fn
    train_losses = []
    test_losses = []
    true_values = []
    predicted_values = []
    num_epochs = epochs

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n-------")
        model.train()
        train_loss_sum = 0.0

        print("Training:")
        for batch, (images, values) in tqdm(enumerate(train_dataloader)):
            images = images.to(device)
            values = values.to(device)
            outputs = model(images)
            loss = criterion(outputs, values)
            train_loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_sum /= len(train_dataloader)
        train_losses.append(train_loss_sum)
        model.eval()
        test_loss_sum = 0.0

        print("Testing:")
        with torch.no_grad():
            for batch, (images, values) in tqdm(enumerate(test_dataloader)):
                images = images.to(device)
                values = values.to(device)
                outputs = model(images)
                loss = criterion(outputs, values)
                test_loss_sum += loss.item()
            test_loss_sum /= len(test_dataloader)
            test_losses.append(test_loss_sum)
        
        print(test_loss_sum + "" + train_loss_sum)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, 'vit_hackru.pt')

test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
test_labels = [path.parent.stem for path in test_data_paths]

def pred_and_store(test_paths, model, transform, class_names, device):
  test_pred_list = []
  for path in tqdm(test_paths):
    pred_dict = {}
    pred_dict["image_path"] = path
    class_name = path.parent.stem
    pred_dict["class_name"] = class_name

    from PIL import Image
    img = Image.open(path)
    transformed_image = transform(img).unsqueeze(0) 
    model.eval()
    with torch.inference_mode():
      pred_logit = model(transformed_image.to(device))
      pred_prob = torch.softmax(pred_logit, dim=1)
      pred_label = torch.argmax(pred_prob, dim=1)
      pred_class = class_names[pred_label.cpu()]
      pred_dict["pred_class"] = pred_class
  
    pred_dict["correct"] = class_name == pred_class
    test_pred_list.append(pred_dict)

  return test_pred_list

if __name__ == '__main__':
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
    results = train_test_step(model=pretrained_vit,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    epochs=1,
                    device=device)
    
    vit_transforms = vit_weights.transforms()
    test_pred_dicts = pred_and_store(test_paths=test_data_paths,
                                 model=pretrained_vit,
                                 transform=vit_transforms,
                                 class_names=class_names,
                                 device=device)

    test_pred_dicts[:1].values()
    