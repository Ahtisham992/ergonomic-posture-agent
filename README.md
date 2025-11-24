---
title: Ergonomic Posture Analyzer
emoji: 🪑
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Ergonomic Posture Assessment Agent (FastAPI - Docker Space)

This Space hosts an AI Agent that analyzes human posture from an uploaded image and returns ergonomic feedback.
The agent is built using:

* **FastAPI**
* **MediaPipe (lightweight posture landmark detection)**
* **OpenCV**
* **NumPy**

It runs inside a Docker container on Hugging Face Spaces and exposes REST API endpoints for analyzing posture and checking service health.

TensorFlow-based deep learning mode is supported in the project code but disabled by default due to memory limits on free Spaces.

---

## 🚀 Running on Hugging Face

This project is deployed as a **Docker-based Space**, meaning Hugging Face builds and runs the container automatically.

The main application is defined in `main.py` and is launched via:

```
uvicorn main:app --host 0.0.0.0 --port 7860
```

The container listens on port **7860**, as required by Spaces.

---

## 🧠 What the Agent Does

* Accepts an uploaded human image
* Detects posture landmarks (shoulders, neck, spine, and others)
* Computes angles and positioning
* Returns ergonomic posture feedback

The output includes:

* **Detected problems** (e.g., forward head posture, rounded shoulders)
* **Confidence estimate**
* **Processed landmark analysis**
* **Optional annotated visualization**

---

## 📂 Project Structure

```
/main.py           # FastAPI application
/requirements.txt  # Python dependencies
/Dockerfile        # Build instructions for HF Spaces
/README.md         # Documentation (this file)
```

---

## ✨ API Endpoints

### 1️⃣ `POST /<AGENT_NAME>`

Analyze an image and return posture assessment.

**Request:**

* Form-data
* Field: `image` (file)

**Example (cURL):**

```
curl -X POST \
  -F "image=@person.jpg" \
  https://huggingface.co/spaces/<your-space>/run
```

**Returns JSON like:**

```json
{
  "status": "success",
  "errors": [],
  "input_info": {...},
  "output_info": {...},
  "refusal": false
}
```

---

### 2️⃣ `GET /health`

Health check endpoint.

**Example:**

```
https://huggingface.co/spaces/<your-space>/health
```

**Response:**

```json
{ "status": "healthy" }
```

---

## ⚙️ Dependencies (`requirements.txt`)

Minimal dependencies recommended:

```
fastapi
uvicorn[standard]
numpy
opencv-python-headless
mediapipe
pydantic
huggingface-hub
```

TensorFlow is intentionally not installed in the free deployment for performance and build success reasons.

---

## 🐳 Docker (Hugging Face Build)

The `Dockerfile` used:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

---

## 🧪 Testing Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run locally:

```
uvicorn main:app --reload --port 7860
```

Open browser:

```
http://localhost:7860/docs
```

---

## ❗ Troubleshooting

### ❌ Build fails

* Check Hugging Face build logs
* Remove large libraries (e.g., TensorFlow)
* Ensure `Dockerfile` uses small base images

### ❌ Space running but returning 502

Make sure:

* Your app **is listening on port 7860**
* YAML block contains `app_port: 7860`
* FastAPI `app` object exists at module level

### 📦 Large model weights

Instead of storing large files in Git:

* Upload models to a Hugging Face Dataset or Model Hub
* Load at runtime using `snapshot_download`

---

## 📜 License

MIT. Feel free to fork, modify, and improve.

---

## 👨‍💻 Author

AI Posture Detection Agent for academic purposes
Deployment target: Hugging Face Spaces (Free Tier)
