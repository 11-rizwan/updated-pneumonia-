import os import cv2 import numpy as np import tensorflow as tf from tensorflow import keras from tensorflow.keras.applications import EfficientNetB5 from tensorflow.keras.preprocessing.image import ImageDataGenerator from tensorflow.keras.layers import Dense, GlobalAveragePooling2D from tensorflow.keras.models import Model from tensorflow.keras.optimizers import Adam from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

Define dataset path

dataset_path = "chest_xray"

Image Preprocessing

img_size = (150, 150) batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory( dataset_path + "/train", target_size=img_size, batch_size=batch_size, class_mode="binary", subset="training" )

val_data = datagen.flow_from_directory( dataset_path + "/train", target_size=img_size, batch_size=batch_size, class_mode="binary", subset="validation" )

Load EfficientNetB5 base model

base_model = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(150, 150, 3)) base_model.trainable = False  # Freeze the base model initially

Add custom layers

x = base_model.output x = GlobalAveragePooling2D()(x) x = Dense(128, activation='relu')(x) out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=out)

Compile model

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

Train the model initially

history = model.fit(train_data, validation_data=val_data, epochs=5)

Unfreeze some top layers for fine-tuning

for layer in base_model.layers[-20:]:  # Unfreezing last 20 layers layer.trainable = True

Recompile model with a lower learning rate

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

Define callbacks

callbacks = [ ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1), EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1) ]

Fine-tune the model

history_fine = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)

Save the fine-tuned model

model.save("pneumonia_model_efficientnetb5_finetuned.h5")

Load the model

model = keras.models.load_model("pneumonia_model_efficientnetb5_finetuned.h5")

def predict_pneumonia(image_path): img = cv2.imread(image_path) img = cv2.resize(img, (150, 150)) img = img / 255.0 img = np.expand_dims(img, axis=0)  # Add batch dimension

prediction = model.predict(img)
return "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"

Example usage

print(predict_pneumonia("path/to/sample_xray.jpg"))

