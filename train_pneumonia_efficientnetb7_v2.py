import tensorflow as tf
import numpy as np
import albumentations as A
import cv2
import os
from keras.applications import EfficientNetB7
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
import tensorflow_addons as tfa
from keras.utils import Sequence

# Enable Mixed Precision for Faster Training
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Paths
TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
IMG_SIZE = (600, 600)  # Higher resolution for EfficientNetB7
BATCH_SIZE = 16  # Reduce batch size to fit large model in memory
EPOCHS = 50
INITIAL_LR = 1e-3

# Data Augmentation using Albumentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CLAHE(clip_limit=2, p=0.3),
    A.CoarseDropout(max_holes=6, max_height=20, max_width=20, p=0.3),
    A.GaussianBlur(p=0.2)
])

# Custom Data Generator
class AugmentedDataGenerator(Sequence):
    def __init__(self, directory, batch_size, img_size, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.class_labels = {"NORMAL": 0, "PNEUMONIA": 1}  # Define class labels
        self.filenames, self.labels = self._load_filenames()

    def _load_filenames(self):
        filenames = []
        labels = []
        for label_name, label in self.class_labels.items():
            class_dir = os.path.join(self.directory, label_name)
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                filenames.append(file_path)
                labels.append(label)
        return filenames, labels

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, index):
        batch_files = self.filenames[index * self.batch_size: (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size: (index + 1) * self.batch_size]

        images = []
        labels = []

        for file, label in zip(batch_files, batch_labels):
            img = cv2.imread(file)
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.augment:
                img = augmentations(image=img)["image"]

            img = img.astype(np.uint8)  # Ensure image is of type uint8
            img = img / 255.0  # Normalize to [0, 1]

            images.append(img)
            labels.append(label)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

# Load Data
train_generator = AugmentedDataGenerator(TRAIN_DIR, BATCH_SIZE, IMG_SIZE, augment=True)
val_generator = AugmentedDataGenerator(VAL_DIR, BATCH_SIZE, IMG_SIZE, augment=False)

# Load EfficientNetB7 Model
base_model = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(600, 600, 3))
base_model.trainable = False  # Freeze base model initially

# Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(1, activation="sigmoid", dtype="float32")(x)

model = Model(inputs=base_model.input, outputs=x)

# Learning Rate Scheduler (Cosine Decay with Warmup)
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
for layer in base_model.layers[:300]:  # Freeze first 300 layers
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
model.save("pneumonia_detection_efficientnetb7.h5")

print("Model training completed and saved as pneumonia_detection_efficientnetb7.h5!")
