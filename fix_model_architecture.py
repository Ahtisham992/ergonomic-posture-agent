"""
Fix Model Architecture JSON for TensorFlow 2.18+ Compatibility
Run this script to fix the BatchNormalization issue
"""

import json
import shutil
from pathlib import Path

print("🔧 FIXING MODEL ARCHITECTURE FOR TF 2.18+")
print("=" * 80)

# Paths
ARCHITECTURE_PATH = "posture_model_tf218_compatible/content/model_architecture.json"
BACKUP_PATH = "posture_model_tf218_compatible/content/model_architecture.json.backup"

# Step 1: Backup original
print("\n1️⃣ Creating backup...")
try:
    shutil.copy(ARCHITECTURE_PATH, BACKUP_PATH)
    print(f"   ✅ Backup created: {BACKUP_PATH}")
except Exception as e:
    print(f"   ❌ Failed to backup: {e}")
    exit(1)

# Step 2: Load and fix architecture
print("\n2️⃣ Loading architecture...")
try:
    with open(ARCHITECTURE_PATH, 'r') as f:
        architecture = json.load(f)
    print("   ✅ Architecture loaded")
except Exception as e:
    print(f"   ❌ Failed to load: {e}")
    exit(1)

# Step 3: Fix BatchNormalization layers
print("\n3️⃣ Fixing BatchNormalization layers...")


def fix_batch_norm_layer(layer_config):
    """Remove problematic mask parameter from BatchNormalization"""
    if layer_config.get('class_name') == 'BatchNormalization':
        # Remove the inbound_nodes that might have mask
        if 'inbound_nodes' in layer_config:
            for node in layer_config['inbound_nodes']:
                if isinstance(node, dict) and 'kwargs' in node:
                    # Remove mask from kwargs
                    node['kwargs'].pop('mask', None)
                elif isinstance(node, list):
                    for item in node:
                        if isinstance(item, dict) and 'kwargs' in item:
                            item['kwargs'].pop('mask', None)
    return layer_config


# Fix Sequential model
if architecture.get('class_name') == 'Sequential':
    layers = architecture.get('config', {}).get('layers', [])

    fixed_count = 0
    for layer in layers:
        # Fix main layer
        if layer.get('class_name') == 'BatchNormalization':
            layer = fix_batch_norm_layer(layer)
            fixed_count += 1

        # Fix nested Functional model (MobileNetV2)
        if layer.get('class_name') == 'Functional':
            nested_layers = layer.get('config', {}).get('layers', [])
            for nested_layer in nested_layers:
                if nested_layer.get('class_name') == 'BatchNormalization':
                    nested_layer = fix_batch_norm_layer(nested_layer)
                    fixed_count += 1

    print(f"   ✅ Fixed {fixed_count} BatchNormalization layers")

# Step 4: Save fixed architecture
print("\n4️⃣ Saving fixed architecture...")
try:
    with open(ARCHITECTURE_PATH, 'w') as f:
        json.dump(architecture, f, indent=2)
    print(f"   ✅ Fixed architecture saved to: {ARCHITECTURE_PATH}")
except Exception as e:
    print(f"   ❌ Failed to save: {e}")
    print(f"   Restoring backup...")
    shutil.copy(BACKUP_PATH, ARCHITECTURE_PATH)
    exit(1)

# Step 5: Verify the fix
print("\n5️⃣ Verifying fix...")
try:
    import tensorflow as tf
    import numpy as np

    # Try loading the model
    with open(ARCHITECTURE_PATH, 'r') as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)
    print("   ✅ Model architecture loads successfully!")

    # Try loading weights
    weights_path = "posture_model_tf218_compatible/content/best_posture_model.weights.h5"
    model.load_weights(weights_path)
    print("   ✅ Weights loaded successfully!")

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("   ✅ Model compiled successfully!")

    # Test prediction
    test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    predictions = model.predict(test_input, verbose=0)
    print(f"   ✅ Test prediction successful! Output shape: {predictions.shape}")

    print("\n" + "=" * 80)
    print("🎉 MODEL ARCHITECTURE FIXED SUCCESSFULLY!")
    print("=" * 80)
    print("\n✅ You can now start your server:")
    print("   uvicorn main:app --reload --port 8001")
    print("\n✅ The model should load in HYBRID mode")

except Exception as e:
    print(f"   ❌ Verification failed: {e}")
    print(f"\n   Restoring backup...")
    shutil.copy(BACKUP_PATH, ARCHITECTURE_PATH)
    print("\n   The issue may require recreating the model in Google Colab")
    print("   See alternative solution below...")

print("\n" + "=" * 80)