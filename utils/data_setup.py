from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict
from utils.util import convert_img_to_tensor
import pandas as pd

class CustomImageDataset(Dataset):
  """Create Custom dataset 
  """
  def __init__(self, data, transform=None, target_transform=None):
      self.img_labels = pd.Series([i[1] for i in data])
      self.images = pd.Series([i[0] for i in data])
      self.transform = transform #Transform on features
      self.target_transform = target_transform #Transforms on labels

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

def get_train_test_per_animal(animal: str,
                              class_name_to_idx : Dict, 
                              seed = 42,
                              test_size = 0.2,
                              train_data_point = 1200
                             ):
  """Download training data per type of animals in the QuickDraw dataset.
  Args:
      animal (str): name of the animal to download
      class_name_to_idx (Dict): A dictionary containing the mapping from str to id
      seed (int, optional): Set random seed for reproducibility. Defaults to 42.
      test_size (float, optional): Choosing test size compare to training size. Defaults to 0.2.
      train_data_point (int, optional):How many data point per class. Defaults to 1200.

  Returns:
      List[List,List]: List containing training data and testing data
      Example:
      get_train_test_per_animal() -> training_data, testing_data
  """
  all_data = np.load(f"data/{animal.lower()}.npy", allow_pickle = True)
  rng = np.random.default_rng(seed=seed)
  all_train,all_test = train_test_split(all_data, test_size = test_size)
  return ([(convert_img_to_tensor(i),class_name_to_idx[animal]) for i in rng.choice(all_train,train_data_point)]),([(convert_img_to_tensor(i),class_name_to_idx[animal]) for i in rng.choice(all_test,int(train_data_point*0.2))])


# def get_train_test_data(class_names: List) -> [List,List]:
#     train_data, test_data = [],[]
#     for animal in class_names:
#         train_animal, test_animal = get_train_test_per_animal(animal)
#         train_data.extend(train_animal)
#         test_data.extend(test_animal)
#     return train_data, test_data

def create_dataloaders(
    train_data: List, 
    test_data: List, 
    train_transforms: transforms.Compose,
    test_transforms: transforms.Compose, 
    batch_size: int = 32,
):
  """Creates training and testing DataLoaders.

  Takes in a training data list and testing data list and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_data: List containing training data
    test_data: PList containing testing data.
    train_transforms: torchvision transforms to perform on training data.
    test_transforms: torchvision transforms to perform on testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.

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
 # Use CustomImageDataset to create dataset(s)    

  train_dataset = CustomImageDataset(data = train_data,transform = train_transforms)
  test_dataset = CustomImageDataset(data = test_data, transform = test_transforms)

    # Turn images into data loaders
  train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)
    
  test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

  return train_dataloader, test_dataloader