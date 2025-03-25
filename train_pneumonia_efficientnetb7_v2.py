import tensorflow as tf
import numpy as np
import cv2
import os
from keras.applications import EfficientNetV2S
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import AdamW
from keras.utils import Sequence

# Paths
TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
IMG_SIZE = (384, 384)  # Optimized for EfficientNetV2-S
BATCH_SIZE = 32  # Higher batch size for efficiency
EPOCHS = 50
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4  # Weight decay for AdamW

# Custom Data Generator (Without Mixed Precision, OpenCV-Compatible)
class DataGenerator(Sequence):
    def __init__(self, directory, batch_size, img_size, shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.class_labels = {"NORMAL": 0, "PNEUMONIA": 1}
        self.filenames, self.labels = self._load_filenames()
        if self.shuffle:
            self.on_epoch_end()

    def _load_filenames(self):
        filenames, labels = [], []
        for label_name, label in self.class_labels.items():
            class_dir = os.path.join(self.directory, label_name)
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                filenames.append(os.path.join(class_dir, file))
                labels.append(label)
        return filenames, labels

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, index):
        batch_files = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        labels = []

        for file, label in zip(batch_files, batch_labels):
            img = cv2.imread(file)
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0  # Normalize & Ensure OpenCV Compatibility

            images.append(img)
            labels.append(label)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.filenames, self.labels))
            np.random.shuffle(combined)
            self.filenames, self.labels = zip(*combined)

# Load Data
train_generator = DataGenerator(TRAIN_DIR, BATCH_SIZE, IMG_SIZE, shuffle=True)
val_generator = DataGenerator(VAL_DIR, BATCH_SIZE, IMG_SIZE, shuffle=False)

# Load EfficientNetV2-S Model
base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(384, 384, 3))
base_model.trainable = False  # Freeze base model initially

# Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)

# Learning Rate Scheduler (Cosine Decay with Warmup)
def cosine_decay_with_warmup(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs * INITIAL_LR
    else:
        return INITIAL_LR * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)

# AdamW Optimizer
optimizer = AdamW(learning_rate=INITIAL_LR, weight_decay=WEIGHT_DECAY)

# Compile Model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
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
for layer in base_model.layers[:150]:  # Freeze first 150 layers
    layer.trainable = False

# Recompile with Lower LR
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=WEIGHT_DECAY),
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
model.save("pneumonia_detection_efficientnetv2s.h5")

print("Model training completed and saved as pneumonia_detection_efficientnetv2s.h5!")
