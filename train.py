import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =====================================================================
# 1. DATA DIRECTORIES SETUP
# =====================================================================
# Apne dataset folder ke sahi path yahan check karke daal dena agar alag hain toh:
DATASET_DIR = "dataset" 
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# =====================================================================
# 2. INTENSE DATA AUGMENTATION (Normal Lungs Ke Variations Badhane Ke Liye)
# =====================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("📁 Loading Training Dataset...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

print("📁 Loading Validation Dataset...")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =====================================================================
# 3. AUTOMATIC CLASS WEIGHTS COMPUTATION (Biasness Khatam Karne Ka Ilaaj)
# =====================================================================
print("⚖️ Calculating Class Weights Matrix to fix Pneumonia Overbias...")
train_labels = train_generator.classes
unique_classes = np.unique(train_labels)

computed_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_labels
)

# Is dictionary se model ko pata chalega ki Normal image pehchanna kitna costly h
weight_dict = {i: computed_weights[i] for i in range(len(computed_weights))}
print(f"Calculated Target Weights: {weight_dict}")

# =====================================================================
# 4. VGG16 TRANSFER LEARNING ARCHITECTURE
# =====================================================================
print("🧠 Initializing VGG16 Architecture Core...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Oxford ke core layers ko freeze kar rahe hain taaki pre-trained knowledge safe rahe
for layer in base_model.layers:
    layer.trainable = False

# Hamara Custom Classification Head (Top Layers)
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Overfitting rokne ke liye neurons drop kiye
output_layer = Dense(1, activation='sigmoid')(x) # Binary evaluation node

model = Model(inputs=base_model.input, outputs=output_layer)

# =====================================================================
# 5. MICRO-LEARNING RATE COMPILATION
# =====================================================================
model.compile(
    optimizer=Adam(learning_rate=0.0001), # Slow stable convergence rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================================
# 6. MODEL TRAINING (With Balanced Weights Passed)
# =====================================================================
EPOCHS = 10
print(f"🚀 Starting Real Deep Learning Training for {EPOCHS} Epochs...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=weight_dict, # <-- Balanced structural weights injected here!
    verbose=1
)

# =====================================================================
# 7. PRODUCTION MODEL PRODUCTION
# =====================================================================
MODEL_NAME = 'pneumonia_model.keras'
model.save(MODEL_NAME)
print(f"🎉 SUCCESS: Real Balanced Model saved as '{MODEL_NAME}'!")
