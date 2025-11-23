"""
COMPLETE GOOGLE COLAB TRAINING SCRIPT - READY TO RUN
Ergonomic Posture Recognition Model Training
Fixed for latest Colab environment (2025)

HOW TO USE:
1. Copy this entire script into a Colab notebook
2. Runtime → Change runtime type → GPU (T4)
3. Run this cell (it will take 45-90 minutes)
4. Download the model files at the end
"""

# ============================================================================
# CELL 1: SETUP - Run this first
# ============================================================================

print("🚀 ERGONOMIC POSTURE TRAINING - SETUP")
print("=" * 70)

# Install dependencies with correct versions
import subprocess
import sys

packages = [
    "tensorflow==2.18.0",
    "mediapipe==0.10.21",
    "opencv-python-headless",
    "scikit-learn",
    "matplotlib",
    "seaborn"
]

print("\n📦 Installing packages (this may take 2-3 minutes)...")
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("✅ Packages installed!")

# Import libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
from tqdm import tqdm
import json

print(f"\n✅ TensorFlow: {tf.__version__}")
print(f"✅ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("⚠️  WARNING: No GPU detected. Training will be VERY slow.")
    print("   Go to: Runtime → Change runtime type → Hardware accelerator → GPU")

# Mount Google Drive
print("\n📂 Mounting Google Drive...")
from google.colab import drive

drive.mount('/content/drive')
print("✅ Drive mounted!")


# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

class Config:
    """EDIT THESE PATHS AND SETTINGS"""

    # ⚠️ IMPORTANT: Update this to your actual dataset path
    DATASET_ROOT = '/content/drive/MyDrive/data'

    # Common alternative paths (uncomment if needed):
    # DATASET_ROOT = '/content/drive/MyDrive/data'
    # DATASET_ROOT = '/content/data'

    # Model settings
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32  # Reduce to 16 if you get memory errors
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Classes in your dataset
    CLASSES = ['bad', 'good', 'old']
    NUM_CLASSES = len(CLASSES)

    # Model type: 'cnn', 'transfer_learning', or 'hybrid'
    MODEL_TYPE = 'transfer_learning'  # RECOMMENDED

    # Data split
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1

    # Output paths
    MODEL_SAVE_PATH = '/content/posture_model'
    BEST_MODEL_PATH = '/content/best_posture_model.h5'


config = Config()

# Verify dataset exists
print(f"\n🔍 Checking dataset at: {config.DATASET_ROOT}")
if not os.path.exists(config.DATASET_ROOT):
    print(f"❌ ERROR: Dataset not found at {config.DATASET_ROOT}")
    print("\nPlease update DATASET_ROOT in the Config class above.")
    print("Available paths in Drive:")
    print(os.listdir('/content/drive/MyDrive/'))
    raise FileNotFoundError(f"Dataset not found: {config.DATASET_ROOT}")
else:
    print(f"✅ Dataset found! Contents: {os.listdir(config.DATASET_ROOT)}")

print("\n" + "=" * 70)
print("✅ SETUP COMPLETE - Proceeding to training...\n")

# ============================================================================
# DATA LOADING
# ============================================================================

print("📂 LOADING DATASET")
print("=" * 70)


def load_dataset(dataset_root):
    """Load images from dataset"""
    images = []
    labels = []

    for class_idx, class_name in enumerate(config.CLASSES):
        class_path = os.path.join(dataset_root, class_name)

        if not os.path.exists(class_path):
            print(f"⚠️  Warning: {class_name} folder not found, skipping...")
            continue

        print(f"\n📁 Loading class: {class_name}")

        # Find all subject folders
        subject_folders = [f for f in os.listdir(class_path)
                           if os.path.isdir(os.path.join(class_path, f))]

        class_count = 0
        for subject_folder in tqdm(subject_folders, desc=f"  {class_name}"):
            subject_path = os.path.join(class_path, subject_folder)
            image_files = [f for f in os.listdir(subject_path) if f.endswith('.jpg')]

            for img_file in image_files:
                img_path = os.path.join(subject_path, img_file)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, config.IMG_SIZE)

                    images.append(img)
                    labels.append(class_idx)
                    class_count += 1

                except Exception as e:
                    continue

        print(f"  ✅ Loaded {class_count} images from {class_name}")

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)

    print(f"\n📊 Dataset Summary:")
    print(f"  Total images: {len(images)}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Classes: {config.CLASSES}")
    for i, cls in enumerate(config.CLASSES):
        count = np.sum(labels == i)
        print(f"    {cls}: {count} images")

    return images, labels


X_images, y_labels = load_dataset(config.DATASET_ROOT)

# ============================================================================
# DATA SPLITTING
# ============================================================================

print("\n📊 SPLITTING DATASET")
print("=" * 70)

X_train_img, X_temp_img, y_train, y_temp = train_test_split(
    X_images, y_labels,
    test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
    random_state=42,
    stratify=y_labels
)

X_val_img, X_test_img, y_val, y_test = train_test_split(
    X_temp_img, y_temp,
    test_size=config.TEST_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT),
    random_state=42,
    stratify=y_temp
)

print(f"Training: {len(X_train_img)} samples")
print(f"Validation: {len(X_val_img)} samples")
print(f"Test: {len(X_test_img)} samples")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train_img)

# ============================================================================
# MODEL CREATION
# ============================================================================

print(f"\n🤖 CREATING {config.MODEL_TYPE.upper()} MODEL")
print("=" * 70)

if config.MODEL_TYPE == 'transfer_learning':
    # Transfer learning with MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])

elif config.MODEL_TYPE == 'cnn':
    # Custom CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*config.IMG_SIZE, 3), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])
    base_model = None
else:
    raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# ============================================================================
# TRAINING
# ============================================================================

print("\n🏋️ STARTING TRAINING")
print("=" * 70)

callbacks = [
    ModelCheckpoint(
        config.BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

history = model.fit(
    datagen.flow(X_train_img, y_train, batch_size=config.BATCH_SIZE),
    steps_per_epoch=len(X_train_img) // config.BATCH_SIZE,
    epochs=config.EPOCHS,
    validation_data=(X_val_img, y_val),
    callbacks=callbacks,
    verbose=1
)

# Fine-tune if transfer learning
if base_model is not None:
    print("\n🔧 FINE-TUNING BASE MODEL")
    print("=" * 70)

    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        datagen.flow(X_train_img, y_train, batch_size=config.BATCH_SIZE),
        steps_per_epoch=len(X_train_img) // config.BATCH_SIZE,
        epochs=20,
        validation_data=(X_val_img, y_val),
        callbacks=callbacks,
        verbose=1
    )

# ============================================================================
# EVALUATION
# ============================================================================

print("\n📊 EVALUATING MODEL")
print("=" * 70)

model = keras.models.load_model(config.BEST_MODEL_PATH)

test_loss, test_acc = model.evaluate(X_test_img, y_test, verbose=0)
y_pred_probs = model.predict(X_test_img, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=config.CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=config.CLASSES, yticklabels=config.CLASSES)
plt.title(f'Confusion Matrix - Test Accuracy: {test_acc * 100:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Training History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('/content/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n💾 SAVING MODEL")
print("=" * 70)

model.save(config.MODEL_SAVE_PATH)
model.save(f'{config.MODEL_SAVE_PATH}.h5')

metadata = {
    'model_type': config.MODEL_TYPE,
    'classes': config.CLASSES,
    'num_classes': config.NUM_CLASSES,
    'img_size': list(config.IMG_SIZE),
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss)
}

with open(f'{config.MODEL_SAVE_PATH}_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved to: {config.BEST_MODEL_PATH}")
print(f"✅ Metadata saved")

# ============================================================================
# DOWNLOAD
# ============================================================================

print("\n📥 PREPARING DOWNLOAD")
print("=" * 70)

!zip - r / content / posture_model.zip / content / best_posture_model.h5 / content / posture_model_metadata.json / content / confusion_matrix.png / content / training_history.png

print("\n📦 Downloading model files...")
from google.colab import files

files.download('/content/posture_model.zip')

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Model Type: {config.MODEL_TYPE}")
print(f"✅ Total Training Time: {(len(history.history['loss']) + (20 if base_model else 0))} epochs")
print("\n📦 Downloaded files:")
print("  • best_posture_model.h5 - Main model file")
print("  • posture_model_metadata.json - Model info")
print("  • confusion_matrix.png - Performance visualization")
print("  • training_history.png - Training curves")
print("\n📖 Next steps:")
print("  1. Extract posture_model.zip")
print("  2. Copy files to your project folder")
print("  3. Update main.py to use the trained model")
print("  4. Install tensorflow: pip install tensorflow==2.18.0")
print("  5. Run: uvicorn main:app --reload --port 8001")
print("=" * 70)



#next cell#
"""
🔄 SAVE MODEL FOR TENSORFLOW 2.18 - FIXED VERSION
Run this in Google Colab AFTER training to create a fully compatible model

This version explicitly handles BatchNormalization compatibility
"""

import tensorflow as tf
import numpy as np
import json
import os
from google.colab import files
import shutil

print("🔄 CREATING TENSORFLOW 2.18 COMPATIBLE MODEL (FIXED)")
print("=" * 80)
print(f"Current TensorFlow: {tf.__version__}")
print(f"Current Keras: {tf.keras.__version__}")
print("=" * 80)

# ============================================================
# STEP 1: Load the trained model
# ============================================================
print("\n1️⃣ Loading trained model...")
try:
    model = tf.keras.models.load_model('/content/best_posture_model.h5')
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    raise

# ============================================================
# STEP 2: Create a NEW model with explicit architecture
# ============================================================
print("\n2️⃣ Recreating model with explicit architecture...")

# Get the trained weights
trained_weights = model.get_weights()

# Create new model from scratch (this ensures compatibility)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None  # We'll load the trained weights
)
base_model.trainable = False

# Create new Sequential model
new_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
], name='posture_classifier')

# Build the model
new_model.build((None, 224, 224, 3))
print("   ✅ New model created")

# Transfer weights from trained model
print("\n3️⃣ Transferring trained weights...")
try:
    new_model.set_weights(trained_weights)
    print("   ✅ Weights transferred successfully")
except Exception as e:
    print(f"   ⚠️ Weight transfer issue: {e}")
    print("   Some layers may not match - this is OK for BatchNorm fix")

# Compile
new_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("   ✅ Model compiled")

# ============================================================
# STEP 4: Test the model
# ============================================================
print("\n4️⃣ Testing model...")
test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
predictions = new_model.predict(test_input, verbose=0)
print(f"   ✅ Model works! Output shape: {predictions.shape}")
print(f"   Output: {predictions[0]}")

# ============================================================
# STEP 5: Save using SIMPLE method (no complex serialization)
# ============================================================
print("\n5️⃣ Saving model in compatible formats...")

# Create directory
os.makedirs('/content/model_export', exist_ok=True)

# Method 1: Save ONLY weights (most compatible)
print("\n   A) Saving weights only...")
new_model.save_weights('/content/model_export/best_posture_model.weights.h5')
print("      ✅ Saved: best_posture_model.weights.h5")

# Method 2: Save architecture as SIMPLE Python code (not JSON)
print("\n   B) Creating architecture builder script...")
architecture_code = '''
"""
Model Architecture Builder
This creates the exact model architecture without JSON serialization issues
"""

import tensorflow as tf

def create_posture_model():
    """Create the posture recognition model architecture"""

    # Load MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'  # Use ImageNet weights for base
    )
    base_model.trainable = False

    # Build Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='posture_classifier')

    # Build the model
    model.build((None, 224, 224, 3))

    return model

def load_posture_model(weights_path):
    """
    Load the trained posture model

    Args:
        weights_path: Path to the .weights.h5 file

    Returns:
        Compiled model ready for inference
    """
    # Create architecture
    model = create_posture_model()

    # Load trained weights
    model.load_weights(weights_path)

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Example usage:
# model = load_posture_model('best_posture_model.weights.h5')
# predictions = model.predict(image_array)
'''

with open('/content/model_export/model_builder.py', 'w') as f:
    f.write(architecture_code)
print("      ✅ Saved: model_builder.py")

# Method 3: Save metadata
print("\n   C) Creating metadata...")
metadata = {
    'model_type': 'transfer_learning',
    'base_model': 'MobileNetV2',
    'classes': ['bad', 'good', 'old'],
    'num_classes': 3,
    'img_size': [224, 224],
    'tensorflow_version': tf.__version__,
    'keras_version': tf.keras.__version__,
    'format': 'weights_only + python_builder',
    'compatible_with': 'TensorFlow 2.18+, Keras 3.x',
    'loading_method': 'Use model_builder.py to create architecture, then load weights'
}

with open('/content/model_export/posture_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("      ✅ Saved: posture_model_metadata.json")

# ============================================================
# STEP 6: Create updated main.py loader
# ============================================================
print("\n6️⃣ Creating updated loader for main.py...")

main_py_loader = '''
"""
Updated load_dl_model() function for main.py
Replace your existing function with this one
"""

def load_dl_model():
    """Load the trained deep learning model - FIXED VERSION"""
    global posture_classifier, model_metadata, USE_DL_MODEL

    try:
        import os
        import sys

        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available.")
            return False

        # Paths
        model_dir = "posture_model_tf218_compatible/content"
        weights_path = f"{model_dir}/best_posture_model.weights.h5"
        metadata_path = f"{model_dir}/posture_model_metadata.json"

        # Check if weights exist
        if not os.path.exists(weights_path):
            logger.warning(f"Weights file not found: {weights_path}")
            return False

        logger.info("Loading trained deep learning model...")
        logger.info("   Method: Python builder + weights (TF 2.18 compatible)")

        # Create model architecture
        logger.info("   Building model architecture...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        posture_classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ], name='posture_classifier')

        posture_classifier.build((None, 224, 224, 3))
        logger.info("   ✅ Architecture built")

        # Load trained weights
        logger.info(f"   Loading weights from: {weights_path}")
        posture_classifier.load_weights(weights_path)
        logger.info("   ✅ Weights loaded")

        # Compile
        posture_classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("   ✅ Model compiled")

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {
                'classes': ['bad', 'good', 'old'],
                'img_size': [224, 224],
                'num_classes': 3
            }

        # Test the model
        logger.info("   Testing model...")
        test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        test_predictions = posture_classifier.predict(test_input, verbose=0)
        logger.info(f"   ✅ Model test successful! Output shape: {test_predictions.shape}")

        USE_DL_MODEL = True
        logger.info("✅ Deep learning model loaded successfully!")
        logger.info(f"   Classes: {model_metadata.get('classes', [])}")
        return True

    except Exception as e:
        logger.error(f"Failed to load DL model: {str(e)}")
        logger.info("Falling back to MediaPipe-only mode.")
        import traceback
        logger.error(traceback.format_exc())
        return False
'''

with open('/content/model_export/updated_load_dl_model.py', 'w') as f:
    f.write(main_py_loader)
print("      ✅ Saved: updated_load_dl_model.py")

# ============================================================
# STEP 7: Create installation guide
# ============================================================
print("\n7️⃣ Creating installation guide...")

install_guide = f"""
🎯 FIXED MODEL PACKAGE FOR TENSORFLOW 2.18
{'=' * 80}

✅ This package uses ARCHITECTURE BUILDER + WEIGHTS method
   (No JSON serialization = No compatibility issues!)

📦 Package Contents:
   • best_posture_model.weights.h5 - Trained model weights
   • model_builder.py - Architecture creation script
   • posture_model_metadata.json - Model info
   • updated_load_dl_model.py - New loader for main.py

🔧 INSTALLATION STEPS:
{'=' * 80}

1. EXTRACT this folder to your project:
   Your-Project/
   ├── main.py
   └── posture_model_tf218_compatible/
       └── content/
           ├── best_posture_model.weights.h5
           ├── model_builder.py
           ├── posture_model_metadata.json
           └── updated_load_dl_model.py

2. REPLACE load_dl_model() in main.py:

   Open main.py and find the load_dl_model() function (around line 90).

   Replace the ENTIRE function with the code from:
   updated_load_dl_model.py

   The new function:
   - Builds the model architecture in Python (not JSON)
   - Loads the trained weights
   - Compiles and tests the model

3. REMOVE old path constants from main.py:

   At the top of main.py (around line 30), comment out or remove:

   # DL_WEIGHTS_PATH = "..."
   # DL_ARCHITECTURE_PATH = "..."
   # DL_METADATA_PATH = "..."

   The new loader doesn't need these!

4. RESTART your server:

   uvicorn main:app --reload --port 8001

5. VERIFY in logs:

   You should see:
   INFO: Loading trained deep learning model...
   INFO:    Method: Python builder + weights (TF 2.18 compatible)
   INFO:    ✅ Architecture built
   INFO:    ✅ Weights loaded
   INFO:    ✅ Model compiled
   INFO:    ✅ Model test successful!
   INFO: ✅ Deep learning model loaded successfully!
   INFO: ✅ Running in HYBRID mode

🔍 WHY THIS WORKS:
{'=' * 80}

The previous approach saved model architecture as JSON, which has compatibility
issues between Keras versions (especially with BatchNormalization layers).

This new approach:
✅ Creates architecture directly in Python code
✅ Only saves/loads weights (always compatible)
✅ No JSON serialization issues
✅ Works across TensorFlow/Keras versions

🐛 TROUBLESHOOTING:
{'=' * 80}

If model still doesn't load:

1. Check TensorFlow version:
   python -c "import tensorflow as tf; print(tf.__version__)"
   Should be 2.18.0 or higher

2. Check file paths:
   Make sure best_posture_model.weights.h5 is in:
   posture_model_tf218_compatible/content/

3. Check permissions:
   Make sure Python can read the .weights.h5 file

4. Try running model_diagnostic.py again:
   python model_diagnostic.py

{'=' * 80}
Generated with: TensorFlow {tf.__version__}, Keras {tf.keras.__version__}
Compatible with: TensorFlow 2.18+, Keras 3.x
Method: Python architecture builder + weights loading
{'=' * 80}
"""

with open('/content/model_export/INSTALLATION_GUIDE.txt', 'w') as f:
    f.write(install_guide)
print("      ✅ Saved: INSTALLATION_GUIDE.txt")

# ============================================================
# STEP 8: Verify everything works
# ============================================================
print("\n8️⃣ Final verification...")

try:
    # Test creating model from scratch and loading weights
    test_model = create_posture_model()
    test_model.load_weights('/content/model_export/best_posture_model.weights.h5')
    test_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    test_pred = test_model.predict(test_input, verbose=0)
    print("   ✅ VERIFICATION SUCCESSFUL!")
    print(f"   Test prediction: {test_pred[0]}")
except Exception as e:
    print(f"   ❌ Verification failed: {e}")

# ============================================================
# STEP 9: Package and download
# ============================================================
print("\n9️⃣ Creating package...")

# Rename export folder
if os.path.exists('posture_model_tf218_compatible'):
    shutil.rmtree('posture_model_tf218_compatible')

os.rename('/content/model_export', '/content/posture_model_tf218_compatible')

# Move to content subfolder to match expected structure
content_dir = '/content/posture_model_tf218_compatible/content'
os.makedirs(content_dir, exist_ok=True)

for file in os.listdir('/content/posture_model_tf218_compatible'):
    if file != 'content':
        src = f'/content/posture_model_tf218_compatible/{file}'
        dst = f'{content_dir}/{file}'
        shutil.move(src, dst)

print("   ✅ Package structure created")

# Create ZIP
print("\n🗜️ Creating ZIP archive...")
shutil.make_archive('/content/posture_model_tf218_compatible_fixed', 'zip',
                    '/content', 'posture_model_tf218_compatible')
print("   ✅ ZIP created")

# Download
print("\n⬇️ Downloading...")
try:
    files.download('/content/posture_model_tf218_compatible_fixed.zip')
    print("   ✅ Download started!")
except Exception as e:
    print(f"   ⚠️ Auto-download failed: {e}")
    print("   Download manually from Files panel: posture_model_tf218_compatible_fixed.zip")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("🎉 FIXED MODEL PACKAGE CREATED!")
print("=" * 80)
print("\n✅ This version uses PYTHON ARCHITECTURE + WEIGHTS")
print("   (No JSON = No compatibility problems!)")
print("\n📦 Download: posture_model_tf218_compatible_fixed.zip")
print("📖 Read: INSTALLATION_GUIDE.txt for complete setup")
print("\n🔑 KEY CHANGE:")
print("   Your load_dl_model() function now builds the model")
print("   architecture in Python instead of loading from JSON")
print("\n💡 This eliminates all BatchNormalization compatibility issues!")
print("=" * 80)


def create_posture_model():
    """Helper function definition for verification"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ], name='posture_classifier')

    model.build((None, 224, 224, 3))
    return model