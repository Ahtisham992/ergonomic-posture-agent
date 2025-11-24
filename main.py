"""
Ergonomic Posture Detection Agent - GUIDE COMPLIANT VERSION
Follows SPM Agent Registry Format Guide Section F
"""

import io
import time
import base64
import logging
import json
import threading
from typing import List, Optional, Dict, Any
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# TensorFlow for trained model
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning features disabled.")

# ==================== Configuration ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GUIDE COMPLIANT: Agent name in lowercase-with-hyphens format
AGENT_NAME = "ergonomic-posture-agent"
AGENT_VERSION = "2.1.0"

# Deep Learning Model Configuration
DL_MODEL_DIR = "posture_model_tf218_compatible/content"
DL_WEIGHTS_PATH = f"{DL_MODEL_DIR}/best_posture_model.weights.h5"
DL_METADATA_PATH = f"{DL_MODEL_DIR}/posture_model_metadata.json"
USE_DL_MODEL = False
MODEL_LOADING = False

# ==================== GUIDE COMPLIANT: Exact Pydantic Models ====================

class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: Role
    content: str


class AgentRequest(BaseModel):
    """GUIDE COMPLIANT: Exact format required"""
    messages: List[Message]


class AgentResponse(BaseModel):
    """GUIDE COMPLIANT: Exact format required"""
    agent_name: str
    status: Status
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# ==================== Global Variables ====================

posture_classifier = None
model_metadata = None
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = None

# ==================== FastAPI App ====================

app = FastAPI(
    title="Ergonomic Posture Agent",
    version=AGENT_VERSION,
    description="AI Worker Agent for posture detection - SPM Guide Compliant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Load Deep Learning Model (Background) ====================

def load_dl_model_background():
    """Load the trained deep learning model in background thread"""
    global posture_classifier, model_metadata, USE_DL_MODEL, MODEL_LOADING

    MODEL_LOADING = True

    try:
        import os

        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available.")
            MODEL_LOADING = False
            return False

        if not os.path.exists(DL_WEIGHTS_PATH):
            logger.warning(f"Weights file not found: {DL_WEIGHTS_PATH}")
            logger.info("Using MediaPipe only mode.")
            MODEL_LOADING = False
            return False

        logger.info("📄 Loading trained deep learning model in background...")

        # Build model architecture
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        posture_classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(224, 224, 3)),
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
        posture_classifier.load_weights(DL_WEIGHTS_PATH)

        posture_classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Load metadata
        if os.path.exists(DL_METADATA_PATH):
            with open(DL_METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {
                'classes': ['bad', 'good', 'old'],
                'img_size': [224, 224],
                'num_classes': 3
            }

        # Warm up model
        test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = posture_classifier.predict(test_input, verbose=0)

        USE_DL_MODEL = True
        MODEL_LOADING = False
        logger.info("✅ Deep learning model loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to load DL model: {str(e)}")
        MODEL_LOADING = False
        USE_DL_MODEL = False
        return False


# ==================== Helper Functions ====================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image array."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise ValueError(f"Invalid base64 image: {str(e)}")


def calculate_angle(point1: tuple, point2: tuple, point3: tuple) -> float:
    """Calculate angle between three points."""
    try:
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        logger.error(f"Error calculating angle: {str(e)}")
        return 0.0


def classify_with_dl_model(image: np.ndarray) -> Dict[str, Any]:
    """Classify posture using trained deep learning model"""
    if not USE_DL_MODEL or posture_classifier is None:
        return None

    try:
        img_size = tuple(model_metadata['img_size'])
        classes = model_metadata['classes']

        # Preprocess image
        img_resized = cv2.resize(image, img_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Predict
        predictions = posture_classifier.predict(img_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        predicted_class = classes[class_idx]

        all_scores = {classes[i]: float(predictions[0][i]) for i in range(len(classes))}

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_scores,
            'method': 'deep_learning'
        }

    except Exception as e:
        logger.error(f"Error in DL classification: {str(e)}")
        return None


def map_dl_class_to_posture_status(dl_class: str) -> str:
    """Map DL classification to posture status"""
    mapping = {
        'good': 'upright',
        'bad': 'slouching',
        'old': 'slouching'
    }
    return mapping.get(dl_class.lower(), 'unknown')


def get_posture_score_from_dl(dl_result: Dict) -> int:
    """Convert DL confidence to posture score"""
    if not dl_result:
        return None

    predicted_class = dl_result['predicted_class']
    confidence = dl_result['confidence']

    if predicted_class == 'good':
        base_score = 90
        score = base_score + int(confidence * 10)
    elif predicted_class == 'bad':
        base_score = 50
        score = base_score - int(confidence * 30)
    else:
        base_score = 60
        score = base_score - int(confidence * 20)

    return max(0, min(100, score))


def analyze_posture_mediapipe(landmarks) -> Dict[str, Any]:
    """Analyze posture from MediaPipe pose landmarks"""
    try:
        # Extract key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate midpoints
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        ear_mid_x = (left_ear.x + right_ear.x) / 2
        ear_mid_y = (left_ear.y + right_ear.y) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2

        # Calculate posture metrics
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        head_forward_distance = ear_mid_x - shoulder_mid_x
        head_shoulder_vertical = abs(ear_mid_y - shoulder_mid_y)

        spine_angle = calculate_angle(
            (ear_mid_x, ear_mid_y),
            (shoulder_mid_x, shoulder_mid_y),
            (hip_mid_x, hip_mid_y)
        )

        lateral_difference = abs(left_shoulder.y - right_shoulder.y)

        # Initialize score and feedback
        posture_score = 100
        issues = []
        posture_status = "upright"

        # Forward head detection
        if head_forward_distance > 0.05:
            severity = head_forward_distance
            if severity > 0.15:
                posture_score -= 40
                issues.append("severe forward head posture")
                posture_status = "slouching"
            elif severity > 0.10:
                posture_score -= 30
                issues.append("forward head posture detected")
                posture_status = "slouching"
            elif severity > 0.07:
                posture_score -= 20
                issues.append("mild forward head posture")
                posture_status = "slouching"
            else:
                posture_score -= 10
                issues.append("slight forward head position")

        # Spine alignment
        if spine_angle < 165:
            severity = 165 - spine_angle
            if severity > 20:
                posture_score -= 35
                issues.append("severe spine misalignment")
                posture_status = "slouching"
            elif severity > 12:
                posture_score -= 25
                issues.append("poor spine alignment")
                posture_status = "slouching"
            elif severity > 6:
                posture_score -= 15
                issues.append("slight spine misalignment")
        elif spine_angle > 195:
            posture_score -= 12
            issues.append("spine leaning backward")

        # Shoulder level
        if shoulder_slope > 0.05:
            posture_score -= 15
            issues.append("uneven shoulders")

        # Lateral tilt
        if lateral_difference > 0.07:
            posture_score -= 18
            if left_shoulder.y < right_shoulder.y:
                issues.append("leaning left")
                if posture_status == "upright":
                    posture_status = "leaning_left"
            else:
                issues.append("leaning right")
                if posture_status == "upright":
                    posture_status = "leaning_right"

        # Rounded shoulders
        if head_forward_distance > 0.08 and spine_angle < 168:
            if "forward head posture" not in str(issues) and "rounded shoulders" not in str(issues):
                issues.append("rounded shoulders")
                posture_score -= 15

        posture_score = max(0, min(100, posture_score))

        return {
            "posture_score": round(posture_score),
            "posture_status": posture_status,
            "issues": issues,
            "metrics": {
                "spine_angle": round(spine_angle, 2),
                "shoulder_slope": round(shoulder_slope, 4),
                "head_forward_distance": round(head_forward_distance, 4),
                "head_shoulder_vertical": round(head_shoulder_vertical, 4)
            },
            "method": "mediapipe"
        }

    except Exception as e:
        logger.error(f"Error in MediaPipe analysis: {str(e)}")
        raise ValueError(f"MediaPipe analysis failed: {str(e)}")


def combine_analysis_results(mediapipe_result: Dict, dl_result: Optional[Dict]) -> Dict[str, Any]:
    """Combine MediaPipe and deep learning results"""

    if not dl_result:
        feedback = generate_feedback(
            mediapipe_result['posture_score'],
            mediapipe_result['issues']
        )
        return {
            'posture_score': mediapipe_result['posture_score'],
            'posture_status': mediapipe_result['posture_status'],
            'feedback': feedback,
            'detected_pose': mediapipe_result['posture_status'],
            'analysis_method': 'mediapipe_only',
            'mediapipe_analysis': mediapipe_result
        }

    # Combine both methods
    mp_score = mediapipe_result['posture_score']
    mp_status = mediapipe_result['posture_status']

    dl_score = get_posture_score_from_dl(dl_result)
    dl_status = map_dl_class_to_posture_status(dl_result['predicted_class'])

    combined_score = int(0.7 * dl_score + 0.3 * mp_score)

    if dl_status == 'slouching' or mp_status == 'slouching':
        final_status = 'slouching'
    elif 'leaning' in mp_status:
        final_status = mp_status
    else:
        final_status = 'upright'

    feedback_parts = []
    dl_class = dl_result['predicted_class']
    dl_conf = dl_result['confidence']
    feedback_parts.append(
        f"AI Classification: {dl_class.upper()} posture detected with {dl_conf:.1%} confidence."
    )

    if mediapipe_result['issues']:
        feedback_parts.append(
            f"Specific issues detected: {', '.join(mediapipe_result['issues'])}."
        )
        feedback_parts.append(get_correction_advice(mediapipe_result['issues']))
    else:
        feedback_parts.append("No significant postural issues detected. Great job!")

    combined_feedback = " ".join(feedback_parts)

    return {
        'posture_score': combined_score,
        'posture_status': final_status,
        'feedback': combined_feedback,
        'detected_pose': final_status,
        'analysis_method': 'hybrid',
        'mediapipe_analysis': mediapipe_result,
        'dl_classification': dl_result,
        'scores': {
            'combined': combined_score,
            'mediapipe': mp_score,
            'deep_learning': dl_score
        }
    }


def generate_feedback(score: int, issues: List[str]) -> str:
    """Generate feedback based on score and issues"""
    if score >= 90:
        return "Excellent posture! Your spine is well-aligned and your head is properly positioned."
    elif score >= 75:
        return f"Good posture with minor issues: {', '.join(issues)}. {get_correction_advice(issues)}"
    elif score >= 60:
        return f"Fair posture. Issues detected: {', '.join(issues)}. {get_correction_advice(issues)}"
    else:
        return f"Poor posture: {', '.join(issues)}. {get_correction_advice(issues)}"


def get_correction_advice(issues: List[str]) -> str:
    """Generate specific correction advice"""
    advice = []
    issues_str = ' '.join(issues)

    if "forward head" in issues_str or "rounded shoulders" in issues_str:
        advice.append("pull your head back and align it with your shoulders")
    if "spine" in issues_str and "misalignment" in issues_str:
        advice.append("straighten your back and sit upright")
    if "uneven shoulders" in issues_str:
        advice.append("level your shoulders")
    if "leaning left" in issues_str or "leaning right" in issues_str:
        advice.append("center your weight and sit straight")

    if not advice:
        return "Maintain proper alignment and take regular breaks."
    return "Try to: " + ", ".join(advice) + "."


def extract_image_from_messages(messages: List[Message]) -> np.ndarray:
    """
    GUIDE COMPLIANT: Extract base64 image from messages content
    The supervisor sends images as base64 strings in message content
    """
    for message in messages:
        content = message.content.strip()

        # Check if content looks like base64 image data
        if content.startswith('data:image') or (len(content) > 100 and not content.startswith('{')):
            try:
                return decode_base64_image(content)
            except Exception as e:
                logger.warning(f"Failed to decode message as base64 image: {str(e)}")
                continue

    raise ValueError("No image found in messages. Please provide image as base64 in message content.")


# ==================== GUIDE COMPLIANT: API Endpoints ====================

@app.get("/health")
async def health_check():
    """
    GUIDE COMPLIANT: Health check endpoint
    Format: {"status": "ok", "agent_name": "...", "ready": true}
    """
    return {
        "status": "ok",
        "agent_name": AGENT_NAME,
        "ready": True,
        "version": AGENT_VERSION,
        "ml_model_loaded": USE_DL_MODEL,
        "ml_model_loading": MODEL_LOADING,
        "analysis_mode": "hybrid" if USE_DL_MODEL else ("loading" if MODEL_LOADING else "mediapipe_only")
    }


@app.post(f"/{AGENT_NAME}", response_model=AgentResponse)
async def posture_agent_endpoint(request: AgentRequest):
    """
    GUIDE COMPLIANT: Main endpoint
    - Accepts AgentRequest with messages
    - Returns AgentResponse with exact format
    - Never crashes, returns error status on failure
    """
    try:
        logger.info(f"Received request with {len(request.messages)} messages")

        # Extract image from messages
        try:
            image = extract_image_from_messages(request.messages)
        except ValueError as e:
            # GUIDE COMPLIANT: Return error status, don't crash
            return AgentResponse(
                agent_name=AGENT_NAME,
                status=Status.ERROR,
                data=None,
                error_message=str(e)
            )

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run MediaPipe pose detection
        logger.info("Running MediaPipe pose detection...")
        results = pose_detector.process(image_rgb)

        if not results.pose_landmarks:
            # GUIDE COMPLIANT: Return error status
            logger.warning("No pose detected")
            return AgentResponse(
                agent_name=AGENT_NAME,
                status=Status.ERROR,
                data=None,
                error_message="No person detected in the image. Please ensure a clear view of upper body."
            )

        # Analyze with MediaPipe
        logger.info("Analyzing posture with MediaPipe...")
        mediapipe_result = analyze_posture_mediapipe(results.pose_landmarks.landmark)

        # Classify with DL model if available
        dl_result = None
        if USE_DL_MODEL:
            logger.info("Classifying with deep learning model...")
            dl_result = classify_with_dl_model(image)

        # Combine results
        logger.info("Combining analysis results...")
        analysis_results = combine_analysis_results(mediapipe_result, dl_result)

        logger.info(f"Analysis complete. Score: {analysis_results['posture_score']}")

        # GUIDE COMPLIANT: Return success response
        return AgentResponse(
            agent_name=AGENT_NAME,
            status=Status.SUCCESS,
            data={
                "message": analysis_results['feedback'],
                "posture_analysis": analysis_results
            },
            error_message=None
        )

    except Exception as e:
        # GUIDE COMPLIANT: Catch all errors, return error status
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return AgentResponse(
            agent_name=AGENT_NAME,
            status=Status.ERROR,
            data=None,
            error_message=f"Internal server error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with agent information"""
    return {
        "agent": AGENT_NAME,
        "version": AGENT_VERSION,
        "status": "operational",
        "ml_model_loaded": USE_DL_MODEL,
        "endpoints": {
            "main": f"/{AGENT_NAME}",
            "health": "/health"
        },
        "description": "Ergonomic Posture Detection Agent - SPM Guide Compliant"
    }


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global pose_detector

    logger.info(f"{AGENT_NAME} v{AGENT_VERSION} starting up...")

    # Initialize MediaPipe
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    logger.info("✅ MediaPipe initialized")

    # Load DL model in background
    logger.info("📄 Starting DL model loading in background...")
    thread = threading.Thread(target=load_dl_model_background, daemon=True)
    thread.start()

    logger.info("✅ Server ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(f"{AGENT_NAME} shutting down...")
    if pose_detector:
        try:
            pose_detector.close()
        except:
            pass


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn

    logger.info("="*60)
    logger.info("🚀 Starting Ergonomic Posture Agent (Guide Compliant)")
    logger.info("="*60)
    logger.info(f"Agent Name: {AGENT_NAME}")
    logger.info(f"Main Endpoint: POST /{AGENT_NAME}")
    logger.info(f"Health Check: GET /health")
    logger.info("Port: 8002 (Guide Compliant)")
    logger.info("="*60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,  # GUIDE COMPLIANT: Changed from 8002 to 8001
        log_level="info"
    )