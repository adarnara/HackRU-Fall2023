#set up 
import torch 
import torchvision
import matplotlib.pyplot as plt 
from torch import nn 
from torchvision import transforms, datasets, models
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
    transform: transforms.ToTensor(), 
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


if __name__ == '__main__':
    random_seed = 123
    num_classes = 23
    learning_rate=1e-3
    torch.manual_seed(random_seed)

    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 23)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms, 
        batch_size=32
    )
    for images, labels in train_dataloader:  
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    results = train_test_step(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    epochs=1,
                    device=device)
    