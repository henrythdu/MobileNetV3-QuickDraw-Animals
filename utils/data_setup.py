from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.img_labels = pd.Series([i[1] for i in data])
        self.images = pd.Series([i[0] for i in data])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images.iloc[idx]
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    


def create_dataloaders(
    train_data: List, 
    test_data: List, 
    train_transforms: transforms.Compose,
    test_transforms: transforms.Compose, 
    batch_size: int = 32,
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  train_dataset = CustomImageDataset(data = train_data,transform = train_transforms)
  test_dataset = CustomImageDataset(data = test_data, transform = test_transforms)


  # Use ImageFolder to create dataset(s)
  
    


    # Turn images into data loaders
  train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)
    
  test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

  return train_dataloader, test_dataloader