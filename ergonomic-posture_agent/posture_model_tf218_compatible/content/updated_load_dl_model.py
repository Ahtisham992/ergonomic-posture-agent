
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
