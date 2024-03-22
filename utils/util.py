import requests
from typing import List, Dict
import torch
from torch import nn
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

def download_data(animal: str):
    """A function to download data from Quick Draw dataset

    Args:
        animal (str):name of the animal
    """
    URL = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{animal.lower()}.npy"
    FILE_TO_SAVE_AS = f"data/{animal.lower()}.npy" # the name to save file as

    resp = requests.get(URL) # making requests to server

    with open(FILE_TO_SAVE_AS, "wb") as f: # opening a file handler to create new file 
        f.write(resp.content) # writing content to file

def load_class_names(file_path: str) -> List:
    """Load class name into a variable

    Args:
        file_path (str): Path to class name text file

    Returns:
        List of all animal names
    """
    with open(file_path, "r") as f:
        class_names_loaded = [animal.strip() for animal in  f.readlines()]
    return class_names_loaded

def convert_class_names_to_idx(class_names: List) -> Dict:
    """Converting the class name of into idx. For example: ["Ant", "Bear"] -> {"Ant":0,"Bear":1}

    Args:
        class_names (List): List of all the class names

    Returns:
        Dict:a dictionary mapping animal name to id
    """
    class_name_to_idx = {}
    for i,animal in enumerate(class_names):
        class_name_to_idx[animal] = i
    return class_name_to_idx

def convert_img_to_tensor(img: np.array) -> torch.Tensor:
    """Converting numpy image into torch tensor

    Args:
        img (np.array): numpy version of the dataset

    Returns:
        torch.Tensor: image in tensor format
    """
    convert_img = torch.from_numpy(img.reshape(28,28))
    convert_img = convert_img.repeat(3, 1, 1)
    return convert_img



def accuracy_tracking_per_batch(output: torch.Tensor, target: torch.Tensor, topk=(1,5)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = [] # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
#             print(f"Total correct this batch is: {tot_correct_topk} out of {batch_size}")
#             topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(tot_correct_topk)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]    
    
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "train_acc_top5: [...]",
             "test_loss": [...],
             "test_acc": [...],
             "test_acc_top5: [...]"}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    accuracy_top5 = results['train_acc_top5']
    test_accuracy = results["test_acc"]
    test_accuracy_top5 = results["test_acc_top5"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, accuracy_top5, label="train_accuracy_top5")
    plt.plot(epochs, test_accuracy_top5, label="test_accuracy_top5")
    plt.title("Accuracy_top5")
    plt.xlabel("Epochs")
    plt.legend()

def save_model(model: torch.nn.Module,
              target_dir: str,
              model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
    either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
              target_dir="models",
              model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                      exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
            f=model_save_path)