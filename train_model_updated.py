import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
import numpy as np
import albumentations as A
import cv2
import os

# Mixed Precision for Faster Training
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
IMG_SIZE = (256, 256)  # Higher resolution for better features
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LR = 1e-3

# Data Augmentation with Albumentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(clip_limit=2, p=0.3),
    A.CoarseDropout(max_holes=6, max_height=20, max_width=20, p=0.3),
    A.GaussianBlur(p=0.2)
])

# Custom Data Generator with Augmentations
class AugmentedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, img_size, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.filenames = [os.path.join(directory, f) for f in os.listdir(directory)]

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, index):
        batch_files = self.filenames[index * self.batch_size: (index + 1) * self.batch_size]
        images, labels = [], []

        for file in batch_files:
            img = cv2.imread(file)
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            
            if self.augment:
                img = augmentations(image=img)["image"]

            label = 1 if "pneumonia" in file.lower() else 0
            images.append(img)
            labels.append(label)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

# Load Data
train_generator = AugmentedDataGenerator(TRAIN_DIR, BATCH_SIZE, IMG_SIZE, augment=True)
val_generator = AugmentedDataGenerator(VAL_DIR, BATCH_SIZE, IMG_SIZE, augment=False)

# Load EfficientNetV2B2 (More Efficient)
base_model = EfficientNetV2B2(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

# Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(1, activation="sigmoid", dtype="float32")(x)  # Label Smoothing Compatible

model = Model(inputs=base_model.input, outputs=x)

# Learning Rate Scheduler with Warmup
def cosine_decay_with_warmup(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs * INITIAL_LR
    else:
        return INITIAL_LR * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)

# Lookahead Optimizer with RAdam
optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(learning_rate=INITIAL_LR))

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
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, lr_scheduler]
)

# Unfreeze Base Model for Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

# Recompile with Lower LR
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fine-Tune the Model
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS // 2,
    callbacks=[early_stopping, lr_scheduler]
)

# Save Model for Deployment
model.save("pneumonia_detection_optimized.h5")

# Convert to TF-Lite for Faster Inference
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("pneumonia_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model optimized and saved as TF-Lite!")
