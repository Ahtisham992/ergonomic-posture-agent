
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
