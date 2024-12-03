import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

import wandb
from wandb.integration.keras import WandbMetricsLogger


# Parameters for data generators
image_size = (224, 224)
batch_size = 32


# Initialize WandB
run = wandb.init(
    project="trash-classification", 
    config={
        "learning_rate": 2e-5,
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 20,
    }
)

config = wandb.config


def generate_data(base_dir):
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.25,
    )

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
    )

    val_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator


def build_model(train_generator):
    # Load pre-trained MobileNetV2
    n_class = len(train_generator.class_indices)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.6)(x)
    output = Dense(n_class, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)

    return model

def compile_train_model(train_generator, val_generator, model):
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss=config.loss,
        metrics=[config.metric]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=config.epoch,
        callbacks=[early_stopping, WandbMetricsLogger()]
    )

    # Save the model locally
    path = "./model/trash-classification-mobilenetv2-finetuned.keras"
    model.save(path)

    # Save model to W&B
    path = "./model/trash-classification-mobilenetv2-finetuned.keras"
    registered_model_name = "trash-classification"

    run.link_model(path=path, registered_model_name=registered_model_name)

    print(f"The model hsa been trained and saved in W&B '{path}'")

    wandb.finish()

    if __name__ == "__main__":
        base_dir = os.path.expanduser('dataset-resized')

        train_generator, val_generator = generate_data(base_dir)

        model = build_model(train_generator)

        compile_train_model(train_generator, val_generator, model)