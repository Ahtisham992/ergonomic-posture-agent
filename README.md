# Ergonomic Posture Agent - Setup and Testing Guide

## 📋 Overview
This is a production-ready FastAPI AI Worker Agent that analyzes ergonomic posture using MediaPipe Pose and OpenCV. It integrates seamlessly with supervisor-worker agent systems and follows strict JSON contracts.

---

## 🚀 Installation Instructions

### Step 1: Create Project Directory
```bash
mkdir ergonomic-posture-agent
cd ergonomic-posture-agent
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Save requirements.txt (provided in artifacts)
# Then install:
pip install -r requirements.txt
```

### Step 4: Save Main Application
Save the `main.py` file (provided in artifacts) to your project directory.

---

## ▶️ Running the Agent

### Start the Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['/path/to/project']
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Ergonomic Posture Agent v1.0.0 starting up...
INFO:     MediaPipe Pose model loaded successfully
INFO:     Server ready to accept requests
INFO:     Application startup complete.
```

The agent is now running at: **http://localhost:8001**

---

## 🧪 Testing the Agent

### Test 1: Health Check
```bash
curl http://localhost:8001/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "agent_name": "Ergonomic Posture Agent",
  "version": "1.0.0",
  "uptime_seconds": 45.23
}
```

---

### Test 2: Register Endpoint
```bash
curl -X POST http://localhost:8001/register
```

**Expected Response:**
```json
{
  "status": "registered",
  "agent_name": "Ergonomic Posture Agent",
  "message": "Agent registered successfully",
  "data": {
    "agent_name": "Ergonomic Posture Agent",
    "version": "1.0.0",
    "capabilities": ["posture_detection", "ergonomic_analysis"],
    "endpoints": {
      "main": "/posture-agent",
      "health": "/health"
    }
  }
}
```

---

### Test 3: Posture Analysis with Image Upload

#### Prepare Test Image
Download or use an image showing a person's upper body. For testing, you can use this sample command:

```bash
# Example with a local image file
curl -X POST http://localhost:8001/posture-agent \
  -F "file=@/path/to/your/image.jpg" \
  -F 'request={"messages":[{"role":"user","content":"Analyze my posture"}]}'
```

**Expected Success Response:**
```json
{
  "agent_name": "Ergonomic Posture Agent",
  "status": "success",
  "data": {
    "posture_score": 82,
    "posture_status": "upright",
    "feedback": "Good posture with minor issues: forward head posture detected. Try to pull your head back to align with your shoulders.",
    "detected_pose": "upright",
    "metrics": {
      "spine_angle": 175.34,
      "shoulder_slope": 0.0234,
      "head_forward_distance": 0.0567,
      "issues": ["forward head posture detected"]
    }
  },
  "error_message": null
}
```

---

### Test 4: Error Handling - No Image
```bash
curl -X POST http://localhost:8001/posture-agent \
  -F 'request={"messages":[{"role":"user","content":"Test without image"}]}'
```

**Expected Error Response:**
```json
{
  "agent_name": "Ergonomic Posture Agent",
  "status": "error",
  "data": null,
  "error_message": "No image found in request. Please provide an image file or base64 data."
}
```

---

### Test 5: Posture Analysis with Base64 Image

```bash
# First, convert an image to base64
base64_image=$(base64 -i /path/to/your/image.jpg | tr -d '\n')

# Send request with base64 in message content
curl -X POST http://localhost:8001/posture-agent \
  -H "Content-Type: multipart/form-data" \
  -F "request={\"messages\":[{\"role\":\"user\",\"content\":\"data:image/jpeg;base64,$base64_image\"}]}"
```

---

## 🧑‍💻 Testing with Python

### Python Test Script
```python
import requests
import json

# Test health endpoint
response = requests.get("http://localhost:8001/health")
print("Health Check:", response.json())

# Test posture analysis with image
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "request": json.dumps({
            "messages": [
                {"role": "user", "content": "Analyze my posture"}
            ]
        })
    }
    response = requests.post(
        "http://localhost:8001/posture-agent",
        files=files,
        data=data
    )
    print("Posture Analysis:", json.dumps(response.json(), indent=2))
```

---

## 📊 Understanding the Response

### Posture Status Values
- **`upright`**: Good posture, minimal issues
- **`slouching`**: Forward head posture or poor spine alignment
- **`leaning_left`**: Tilting to the left side
- **`leaning_right`**: Tilting to the right side

### Posture Score Ranges
- **85-100**: Excellent posture
- **70-84**: Good posture with minor issues
- **50-69**: Fair posture, needs correction
- **0-49**: Poor posture, immediate correction needed

### Common Issues Detected
- Forward head posture
- Poor spine alignment
- Uneven shoulders
- Lateral leaning (left/right)

---

## 🔧 Troubleshooting

### Issue: MediaPipe Installation Error
```bash
# Try installing with specific version
pip install mediapipe==0.10.8 --no-cache-dir
```

### Issue: OpenCV Import Error
```bash
# Install opencv-python-headless for server environments
pip install opencv-python-headless
```

### Issue: Port Already in Use
```bash
# Use a different port
uvicorn main:app --reload --port 8002
```

---

## 🏗️ Integration with Supervisor

To integrate with a supervisor agent:

1. Configure `SUPERVISOR_URL` in main.py
2. The supervisor can call `/register` to register this agent
3. Route requests to `/posture-agent` endpoint
4. Parse `AgentResponse` for results

---

## 📝 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## ✅ Production Checklist

- [x] Follows exact AgentRequest/AgentResponse schema
- [x] MediaPipe Pose integration
- [x] OpenCV image processing
- [x] Comprehensive error handling
- [x] Logging for debugging
- [x] Health check endpoint
- [x] Registration endpoint
- [x] Detailed posture metrics
- [x] User-friendly feedback
- [x] Ready for supervisor integration

---

## 📚 Additional Notes

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- Base64 encoded images

### Performance Tips
- Images between 640x480 and 1920x1080 work best
- Ensure good lighting for accurate detection
- Full upper body should be visible
- Face the camera directly for best results

### Security Considerations
- Add authentication for production
- Implement rate limiting
- Validate image file sizes
- Sanitize user inputs

---

## 🎓 University Project Requirements Met

✅ FastAPI-based AI Worker Agent  
✅ Strict JSON request/response contracts  
✅ MediaPipe + OpenCV integration  
✅ Supervisor-worker architecture ready  
✅ Health and registration endpoints  
✅ Complete error handling  
✅ Production-ready code quality  
✅ Comprehensive documentation  
✅ Ready to run immediately  

---

**Agent Status**: ✅ Fully Operational  
**Version**: 1.0.0  
**Last Updated**: 2025