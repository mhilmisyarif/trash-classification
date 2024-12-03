# Automated Trash Classification

This repository is contains experiment to build Fine-Tuned Model based on `MobileNetV2` to classify the trash based on the images data available in this [Hugging Face Repository](https://huggingface.co/datasets/garythung/trashnet). This model is built using the `dataset-resized` datasets for limited training computing.

## Project Structures

The project is publised in 3 main files:

1. `Development and Experimentation.ipynb`: This notebook contains initial exploration of the data and experiment to develop the model.
2. `train_model.py`: This python script automates the model development.

## Automated Training

The training script `train_model.py` will run automtically at every 12AM. The detailed automation script is written in `.github/workflows/main.yaml` by using `WANDB_API_KEY` as a secret variable

## How to Reproduce

To replicate the experiment process you will need to:

1. Clone the repository,

```bash
git clone <github-repository-url>
```

2. Create environment using `python3.12.x`. In this case I use venv,

```bash
python -m venv venv
```

3. Install required dependencies,

```bash
pip install -r requirements.txt
```

4. Create `model` directory,

```bash
mkdir -p `./model`
```

5. Run `train_model.py`,

```bash
python train_model.py
```

## Pre-trained Model

The pre-trained model is available to use on the [Huggingface Repository](https://huggingface.co/hilmisyarif/trash-classification-mobilenetv2-finetuned)
