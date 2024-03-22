# QuickDraw Animal Predictor with MobileNetV3

## Overview
This project utilizes the power of transfer learning to adapt the MobileNetV3 model for the task of predicting animal drawings from the QuickDraw dataset. By leveraging the pre-trained weights of MobileNetV3, I fine-tune the model to recognize various animals as drawn by users around the world.

## Dataset
The QuickDraw dataset is a collection of 50 million drawings across 345 categories, contributed by players of the game "Quick, Draw!". For this project, we focus on the subset of categories that represent animals.

## Model
MobileNetV3 is a state-of-the-art lightweight deep neural network designed for mobile and edge devices. It's efficient and fast, making it ideal for real-time applications.

## Requirements
torch==2.1.2
torchvision==0.16.2
tqdm==4.66.1
matplotlib==3.7.5
numpy==1.26.4
pandas==2.2.0
sklearn==1.2.2
torchinfo==1.8.0

## Installation
To set up the project environment, run the following commands:
```bash
git clone https://github.com/henrythdu/MobileNetV3-QuickDraw-Animals.git
cd MobileNetV3-QuickDraw-Animals
pip install -r requirements.txt
```
## Usage
To train the model with the QuickDraw dataset, run:
```
MobileNetV3 model.ipynb
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
## Acknowledgments

Google’s Quick, Draw! team for providing the dataset.

