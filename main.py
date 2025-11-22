"""
Ergonomic Posture Detection Agent
A FastAPI-based AI Worker Agent that analyzes user posture using MediaPipe Pose and OpenCV.
Follows strict Agent Format Guide specifications.
"""

import io
import time
import base64
import logging
from typing import List, Optional, Dict, Any
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ==================== Configuration ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agent metadata - MUST match format guide
AGENT_NAME = "ergonomic-posture-agent"  # FIXED: lowercase with dashes
AGENT_VERSION = "1.0.0"

# Track uptime
START_TIME = time.time()

# ==================== Pydantic Models (Format Guide Compliant) ====================

class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Message format as specified in Format Guide"""
    role: Role
    content: str


class AgentRequest(BaseModel):
    """EXACT format from Format Guide - DO NOT MODIFY"""
    messages: List[Message]


class AgentResponse(BaseModel):
    """EXACT format from Format Guide - DO NOT MODIFY"""
    agent_name: str
    status: Status
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# ==================== MediaPipe Initialization ====================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# ==================== FastAPI App ====================

app = FastAPI(
    title="Ergonomic Posture Agent",
    version=AGENT_VERSION,
    description="AI Worker Agent for ergonomic posture detection and analysis"
)


# ==================== Helper Functions ====================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image array."""
    try:
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


def analyze_posture(landmarks) -> Dict[str, Any]:
    """Analyze posture from MediaPipe pose landmarks."""
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

        # Spine alignment checking
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

        # Check shoulder level
        if shoulder_slope > 0.05:
            posture_score -= 15
            issues.append("uneven shoulders")

        # Check lateral tilt
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

        # Check for rounded shoulders
        if head_forward_distance > 0.08 and spine_angle < 168:
            if "forward head posture" not in str(issues) and "rounded shoulders" not in str(issues):
                issues.append("rounded shoulders")
                posture_score -= 15

        posture_score = max(0, min(100, posture_score))

        # Generate feedback
        if posture_score >= 90:
            feedback = "Excellent posture! Your spine is well-aligned and your head is properly positioned."
        elif posture_score >= 75:
            feedback = f"Good posture with minor issues: {', '.join(issues)}. {get_correction_advice(issues)}"
        elif posture_score >= 60:
            feedback = f"Fair posture. Issues detected: {', '.join(issues)}. {get_correction_advice(issues)}"
        else:
            feedback = f"Poor posture: {', '.join(issues)}. {get_correction_advice(issues)}"

        logger.info(f"Posture Score: {posture_score}, Head Forward: {head_forward_distance:.4f}, Spine Angle: {spine_angle:.2f}")

        return {
            "posture_score": round(posture_score),
            "posture_status": posture_status,
            "feedback": feedback,
            "detected_pose": posture_status,
            "metrics": {
                "spine_angle": round(spine_angle, 2),
                "shoulder_slope": round(shoulder_slope, 4),
                "head_forward_distance": round(head_forward_distance, 4),
                "head_shoulder_vertical": round(head_shoulder_vertical, 4),
                "issues": issues
            }
        }

    except Exception as e:
        logger.error(f"Error analyzing posture: {str(e)}")
        raise ValueError(f"Posture analysis failed: {str(e)}")


def get_correction_advice(issues: List[str]) -> str:
    """Generate specific correction advice based on detected issues."""
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


def process_image(image_data: bytes) -> np.ndarray:
    """Process uploaded image bytes to OpenCV format."""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Invalid image format: {str(e)}")


def extract_image_from_request(request: AgentRequest, file: Optional[UploadFile]) -> np.ndarray:
    """Extract image from either uploaded file or base64 content in messages."""
    # Try uploaded file first
    if file is not None:
        try:
            image_data = file.file.read()
            return process_image(image_data)
        except Exception as e:
            logger.error(f"Error reading uploaded file: {str(e)}")
            raise ValueError(f"Failed to read uploaded file: {str(e)}")

    # Try to extract base64 from messages
    for message in request.messages:
        content = message.content.strip()
        if content.startswith('data:image') or len(content) > 100:
            try:
                return decode_base64_image(content)
            except Exception as e:
                logger.warning(f"Failed to decode message content as base64: {str(e)}")
                continue

    raise ValueError("No image found in request. Please provide an image file or base64 data.")


# ==================== API Endpoints (Format Guide Compliant) ====================

@app.get("/health")
async def health_check():
    """
    Health check endpoint - REQUIRED by Format Guide
    Must return: status, agent_name, ready
    """
    return {
        "status": "ok",
        "agent_name": AGENT_NAME,
        "ready": True
    }


@app.post("/ergonomic-posture-agent", response_model=AgentResponse)
async def posture_agent_endpoint(
    request: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Main endpoint - MUST match format: POST /ergonomic-posture-agent
    MUST accept AgentRequest format
    MUST return AgentResponse format
    """
    try:
        # Parse request JSON
        if request is None:
            agent_request = AgentRequest(messages=[
                Message(role=Role.USER, content="Analyze my posture")
            ])
        else:
            try:
                import json
                request_dict = json.loads(request)
                agent_request = AgentRequest(**request_dict)
            except Exception as e:
                logger.error(f"Error parsing request JSON: {str(e)}")
                # FIXED: Return error in correct format
                return AgentResponse(
                    agent_name=AGENT_NAME,
                    status=Status.ERROR,
                    data=None,
                    error_message=f"Invalid request format: {str(e)}"
                )

        logger.info(f"Received request with {len(agent_request.messages)} messages")

        # Extract image
        try:
            image = extract_image_from_request(agent_request, file)
        except ValueError as e:
            # FIXED: Return error in correct format
            return AgentResponse(
                agent_name=AGENT_NAME,
                status=Status.ERROR,
                data=None,
                error_message=str(e)
            )

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect pose
        logger.info("Running MediaPipe pose detection...")
        results = pose_detector.process(image_rgb)

        if not results.pose_landmarks:
            logger.warning("No pose detected")
            # FIXED: Return error in correct format
            return AgentResponse(
                agent_name=AGENT_NAME,
                status=Status.ERROR,
                data=None,
                error_message="No person detected in the image. Please ensure a clear view of upper body."
            )

        # Analyze posture
        logger.info("Analyzing posture...")
        analysis_results = analyze_posture(results.pose_landmarks.landmark)

        logger.info(f"Analysis complete. Score: {analysis_results['posture_score']}")

        # FIXED: Return success in correct format with "message" key
        return AgentResponse(
            agent_name=AGENT_NAME,
            status=Status.SUCCESS,
            data={
                "message": analysis_results['feedback'],  # Primary message
                "posture_analysis": analysis_results  # Full analysis data
            },
            error_message=None
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        # FIXED: Return error in correct format
        return AgentResponse(
            agent_name=AGENT_NAME,
            status=Status.ERROR,
            data=None,
            error_message=f"Internal server error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with agent information."""
    return {
        "agent": AGENT_NAME,
        "version": AGENT_VERSION,
        "status": "operational",
        "endpoints": {
            "main": f"/{AGENT_NAME}",
            "health": "/health"
        },
        "description": "AI Worker Agent for ergonomic posture detection and analysis"
    }


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    logger.info(f"{AGENT_NAME} v{AGENT_VERSION} starting up...")
    logger.info("MediaPipe Pose model loaded successfully")
    logger.info("Server ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{AGENT_NAME} shutting down...")
    pose_detector.close()


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )