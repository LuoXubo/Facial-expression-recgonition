# Facial-expression-recgonition

## Introduction

This project is a facial expression recognition framework that can recognize the facial expressions of a singal person or servel persons. The framework can recognize seven different facial expressions: anger, disgust, fear, happiness, sadness, surprise, and neutral. We select the FER2013 dataset as the training dataset, and some popular models like ResNet family, Transformer, and Swin Transformer are evaluated on the validation dataset.

## Dataset

The FER2013 dataset is used in this project. The dataset contains 35,887 grayscale, 48x48 sized face images with seven different facial expressions: anger, disgust, fear, happiness, sadness, surprise, and neutral. The dataset is split into three parts: training, validation, and test. The training set contains 28,709 samples, the validation set contains 3,589 samples, and the test set contains 3,589 samples.

## Models

The following models are evaluated on the validation dataset:

- ResNet50
- ResNet101
- ResNet152
- Transformer
- Swin Transformer
- EfficientNetB0
- EfficientNetB4
- EfficientNetB7

## Results

The following table shows the accuracy and timecost of the models on the validation dataset:

Dataset: FER2013
| Model | Accuracy | Timecost |
| ---------------- | -------- | -------- |
| ResNet50 | 0.9753 | 0.25 |
| ResNet101 | 0.9811 | 0.35 |
| ResNet152 | 0.9713 | 0.45 |
| Transformer | 0.68 | 0.55 |
| Swin Transformer | 0.69 | 0.65 |

Dataset: RAF-DB
| Model | Accuracy | Timecost |
| ---------------- | -------- | -------- |
| ResNet50 | 0.8067 | 0.1138 |
| ResNet101 | 0.7465 | 0.2479 |
| ResNet152 | 0.7803 | 0.3358 |
| EfficientNetB0 | 0.8136 | 0.75 |
| EfficientNetB4 | 0.7939 | 0.75 |
| EfficientNetB7 | 0.8029 | 0.75 |
| Vision Transformer | 0.6578 | 0.1202 |

## Usage

To train the model, run the following command:

```bash
python train.py
```

To evaluate the model, run the following command:

```bash
python evaluate.py
```

To recognize the facial expressions of a single person or several persons, run the following command:

```bash
python demo.py --image_path path/to/image
```

To quickly test the model, you can run the demo.ipynb notebook.

![Some samples predicted by ResNet50.](./samples.jpg)

## Requirements

- Python 3.8
- PyTorch 1.9.0
- torchvision 0.10.0
- numpy 1.21.0
- opencv-python 4.5.2
- matplotlib 3.4.2
- pandas 1.3.0
- scikit-learn 0.24.2
- albumentations 1.0.3
- timm 0.4.12
- torchinfo 0.1.5

## References

- [FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [Transformer](https://arxiv.org/abs/1706.03762)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [EfficientNet](https://arxiv.org/abs/1905.11946)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
