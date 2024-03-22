import torch
import torchvision

from torch import nn

def create_mobilenetv3_model(num_classes:int=42, 
                             seed:int=42):
    """Creates an MobileNetv2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 42.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): MobileNetv2 feature extractor model. 
        transforms (torchvision.transforms): MobileNetv2 image transforms.
    """
    # Create MobileNetv2 pretrained weights, transforms and model
    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.mobilenet_v3_large(weights = weights)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=960, out_features=1280, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes, bias=True)
                            )

    return model, transforms