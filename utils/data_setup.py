from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List
from utils.util import convert_img_to_tensor
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

def get_train_test_per_animal(animal: np.array,
                              class_name_to_idx : List, 
                              seed = 42,
                              test_size = 0.2,
                              train_data_point = 1200
                             ):
    all_data = np.load(f"data/{animal.lower()}.npy", allow_pickle = True)
    rng = np.random.default_rng(seed=seed)
    all_train,all_test = train_test_split(all_data, test_size = test_size)
    return ([(convert_img_to_tensor(i),class_name_to_idx[animal]) for i in rng.choice(all_train,train_data_point)]),([(convert_img_to_tensor(i),class_name_to_idx[animal]) for i in rng.choice(all_test,int(train_data_point*0.2))])


def get_train_test_data(class_names: List) -> [List,List]:
    train_data, test_data = [],[]
    for animal in class_names:
        train_animal, test_animal = get_train_test_per_animal(animal)
        train_data.extend(train_animal)
        test_data.extend(test_animal)
    return train_data, test_data

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