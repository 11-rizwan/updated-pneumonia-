import tensorflow as tf
import numpy as np
import albumentations as A
import cv2
import os
from keras.applications import EfficientNetB5
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import AdamW  # Replacing TFA optimizers

# Paths & Hyperparameters
TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
IMG_SIZE = (380, 380)  # Suitable size for B5 on RTX 2050
BATCH_SIZE = 16
EPOCHS = 50
INITIAL_LR = 5e-4  # Optimized learning rate

# Data Augmentation using Albumentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(clip_limit=2, p=0.3),
    A.CoarseDropout(max_holes=6, max_height=20, max_width=20, p=0.3),
    A.GaussianBlur(p=0.2)
])

# TFRecord Data Pipeline for Faster Data Loading
def parse_image(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Ensure proper format for Albumentations
    return img, label

def augment_image(img, label):
    """Apply Albumentations and ensure dtype is correct."""
    img = img.numpy()  # Convert Tensor to NumPy array
    img = (img * 255).astype(np.uint8)  # Convert back to uint8 for Albumentations
    img = augmentations(image=img)["image"]  # Apply augmentations
    img = img.astype(np.float32) / 255.0  # Convert back to float32
    return img, label

def load_dataset(directory, batch_size, augment=False):
    filenames, labels = [], []
    for class_name, class_label in {"NORMAL": 0, "PNEUMONIA": 1}.items():
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue
        for file in os.listdir(class_dir):
            filenames.append(os.path.join(class_dir, file))
            labels.append(class_label)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(lambda x, y: tf.py_function(augment_image, [x, y], [tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Load Data Using Optimized Pipeline
train_dataset = load_dataset(TRAIN_DIR, BATCH_SIZE, augment=True)
val_dataset = load_dataset(VAL_DIR, BATCH_SIZE, augment=False)

# Load EfficientNetB5 Model
base_model = EfficientNetB5(weights="imagenet", include_top=False, input_shape=(380, 380, 3))
base_model.trainable = False  # Freeze base model initially

# Optimized Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="sigmoid")(x)  # Ensure float32 output

model = Model(inputs=base_model.input, outputs=x)

# Learning Rate Scheduler
def cosine_decay_with_warmup(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs * INITIAL_LR
    else:
        return INITIAL_LR * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)

# Updated Optimizer: AdamW
optimizer = AdamW(learning_rate=INITIAL_LR, weight_decay=1e-4)  # Adaptive Weight Decay

# Compile Model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, lr_scheduler]
)

# Unfreeze Base Model for Fine-Tuning
base_model.trainable = True
for layer in base_model.layers:
    if "batch_normalization" in layer.name:
        layer.trainable = False  # Keep BatchNorm layers frozen

# Recompile with Lower LR
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fine-Tune the Model
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS // 2,
    callbacks=[early_stopping, lr_scheduler]
)

# Save Model for Deployment
model.save("pneumonia_detection_efficientnetb5_tf219.h5")

print("Optimized model training completed and saved as pneumonia_detection_efficientnetb5_tf219.h5!")
