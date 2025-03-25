import tensorflow as tf
import numpy as np
import albumentations as A
import cv2
import os
from keras.applications import EfficientNetB5
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import AdamW
from sklearn.utils.class_weight import compute_class_weight

# Disable Mixed Precision
# tf.keras.mixed_precision.set_global_policy("mixed_float16")  # Removed

# Paths & Hyperparameters
TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
IMG_SIZE = (380, 380)
BATCH_SIZE = 16
EPOCHS = 50
INITIAL_LR = 3e-4  # Reduced initial LR
FINE_TUNE_LR = 5e-5  # Lower LR for fine-tuning

# **Improved Data Augmentation**
augmentations = A.Compose([
    A.RandomResizedCrop(height=380, width=380, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_REFLECT, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    A.CLAHE(clip_limit=4, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.CoarseDropout(max_holes=6, max_height=30, max_width=30, p=0.3),
])

# **TFRecord Data Pipeline**
def parse_image(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0,1] range
    return img, label

def augment_image(image, label):
    img = tf.numpy_function(func=lambda img: augmentations(image=img)["image"], inp=[image], Tout=tf.float32)
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
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Load Data
train_dataset = load_dataset(TRAIN_DIR, BATCH_SIZE, augment=True)
val_dataset = load_dataset(VAL_DIR, BATCH_SIZE, augment=False)

# **Compute Class Weights for Balanced Training**
labels = [0] * len(os.listdir(f"{TRAIN_DIR}/NORMAL")) + [1] * len(os.listdir(f"{TRAIN_DIR}/PNEUMONIA"))
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# **Load EfficientNetB5 Model**
base_model = EfficientNetB5(weights="imagenet", include_top=False, input_shape=(380, 380, 3))
base_model.trainable = False  # Freeze base model initially

# **Optimized Custom Layers**
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="sigmoid")(x)  # Ensure float32 output

model = Model(inputs=base_model.input, outputs=x)

# **Cosine Decay Learning Rate Scheduler**
def cosine_decay_with_warmup(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs * INITIAL_LR
    else:
        return INITIAL_LR * 0.2 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)

# **Compile Model**
model.compile(
    optimizer=AdamW(learning_rate=INITIAL_LR, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

# **Callbacks**
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# **Initial Training with Frozen Base Model**
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,  # Train base model first
    callbacks=[early_stopping, lr_scheduler],
    class_weight=class_weights_dict
)

# **Unfreeze Last 50 Layers for Fine-Tuning**
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Freeze the first N layers
    layer.trainable = False  

# **Recompile with Lower LR for Fine-Tuning**
model.compile(
    optimizer=AdamW(learning_rate=FINE_TUNE_LR, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

# **Fine-Tune the Model**
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS - 10,  # Remaining epochs
    callbacks=[early_stopping, lr_scheduler],
    class_weight=class_weights_dict
)

# **Save Model**
model.save("pneumonia_detection_efficientnetb5_optimized.h5")

print("Optimized model training completed and saved as pneumonia_detection_efficientnetb5_optimized.h5!")
