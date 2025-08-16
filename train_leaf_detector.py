from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Parameters
DATASET_DIR = 'C:\Users\shrey\OneDrive\Desktop\leaf_dataset\tomato_leaf'
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = 'leaf_detector_model.h5'

# Data setup with 80/20 train/validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Model architecture
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base and compile
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train and save
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save(MODEL_PATH)
print(f"Saved binary leaf detector model to {MODEL_PATH}")
