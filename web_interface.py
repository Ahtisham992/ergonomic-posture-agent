"""
Web Interface for Ergonomic Posture Agent
Beautiful Gradio-based UI for posture analysis
FIXED VERSION - Works with hybrid mode and displays all metrics
"""

import gradio as gr
import requests
import json
import cv2
import numpy as np
from PIL import Image
import io

# Configuration
AGENT_URL = "http://localhost:8002/ergonomic-posture-agent"
HEALTH_URL = "http://localhost:8002/health"


def check_agent_status():
    """Check if the posture agent is running"""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            agent_name = data.get('agent_name', 'unknown')
            ready = data.get('ready', False)
            ml_loaded = data.get('ml_model_loaded', False)
            mode = data.get('analysis_mode', 'unknown')

            if ready:
                status = f"✅ Agent is running: {agent_name}\n"
                status += f"   Mode: {mode.upper()}\n"
                if ml_loaded:
                    status += "   🤖 ML Model: Loaded"
                else:
                    status += "   ⚠️  ML Model: Not loaded (MediaPipe only)"
                return True, status
            else:
                return False, f"⚠️ Agent not ready: {agent_name}"
        return False, "⚠️ Agent responded with unexpected status"
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to agent. Please start it with:\n   uvicorn main:app --reload --port 8001"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def analyze_posture_image(image):
    """Analyze posture from uploaded image"""
    if image is None:
        return "❌ Please upload an image first", None, ""

    # Check if agent is running
    is_running, status_msg = check_agent_status()
    if not is_running:
        return status_msg + "\n\nPlease start the agent first!", None, ""

    try:
        # Convert image to bytes
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Convert to JPEG bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Prepare request
        files = {
            'file': ('posture.jpg', img_byte_arr, 'image/jpeg')
        }

        data = {
            'request': json.dumps({
                "messages": [
                    {"role": "user", "content": "Analyze my posture"}
                ]
            })
        }

        # Send to agent
        print("📤 Sending request to agent...")
        response = requests.post(AGENT_URL, files=files, data=data, timeout=30)

        print(f"📥 Received response: Status {response.status_code}")

        if response.status_code != 200:
            return f"❌ Error: Server returned status {response.status_code}\n{response.text}", None, ""

        result = response.json()
        print(f"📊 Response data: {json.dumps(result, indent=2)}")

        # Parse results
        if result.get("status") == "error":
            error_msg = result.get("error_message", "Unknown error")
            return f"❌ Analysis Error:\n{error_msg}", None, ""

        if result.get("status") == "success":
            data_obj = result.get("data", {})

            # Extract posture analysis
            posture_analysis = data_obj.get("posture_analysis", {})
            main_message = data_obj.get("message", "No feedback available")

            # Extract information with robust fallbacks
            score = posture_analysis.get("posture_score", 0)
            status = posture_analysis.get("posture_status", "unknown")
            feedback = posture_analysis.get("feedback", main_message)

            # Get metrics - handle both dict and missing cases
            metrics = posture_analysis.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}

            # Get issues - check multiple possible locations
            issues = []
            if "issues" in metrics and isinstance(metrics.get("issues"), list):
                issues = metrics["issues"]
            elif "issues" in posture_analysis and isinstance(posture_analysis.get("issues"), list):
                issues = posture_analysis["issues"]

            # Get mediapipe analysis if available
            mp_analysis = posture_analysis.get("mediapipe_analysis", {})
            if mp_analysis and isinstance(mp_analysis, dict):
                mp_metrics = mp_analysis.get("metrics", {})
                if isinstance(mp_metrics, dict) and not metrics:
                    metrics = mp_metrics
                if "issues" in mp_analysis and not issues:
                    issues = mp_analysis.get("issues", [])

            # Create detailed result text
            result_text = f"""
🎯 POSTURE ANALYSIS RESULTS
{'='*50}

📊 POSTURE SCORE: {score}/100

📍 STATUS: {status.upper()}

💬 FEEDBACK:
{feedback}

📏 DETAILED METRICS:
• Spine Angle: {metrics.get('spine_angle', 'N/A')}°
• Shoulder Slope: {metrics.get('shoulder_slope', 'N/A')}
• Head Forward Distance: {metrics.get('head_forward_distance', 'N/A')}
"""

            if metrics.get('head_shoulder_vertical'):
                result_text += f"• Head-Shoulder Vertical: {metrics.get('head_shoulder_vertical', 'N/A')}\n"

            if issues and len(issues) > 0:
                result_text += f"\n⚠️ ISSUES DETECTED:\n"
                for issue in issues:
                    result_text += f"  • {issue}\n"
            else:
                result_text += f"\n✅ NO MAJOR ISSUES DETECTED!\n"

            # Check if hybrid mode and add ML details
            analysis_method = posture_analysis.get("analysis_method", "unknown")
            if analysis_method == "hybrid":
                dl_class = posture_analysis.get("dl_classification", {})
                if dl_class and isinstance(dl_class, dict):
                    result_text += f"\n🤖 AI CLASSIFICATION:\n"
                    result_text += f"  • Predicted Class: {dl_class.get('predicted_class', 'N/A').upper()}\n"
                    result_text += f"  • Confidence: {dl_class.get('confidence', 0)*100:.1f}%\n"

                    # Show all probabilities
                    all_probs = dl_class.get('all_probabilities', {})
                    if all_probs:
                        result_text += f"  • All Probabilities:\n"
                        for cls, prob in all_probs.items():
                            result_text += f"    - {cls}: {prob*100:.1f}%\n"

                scores = posture_analysis.get("scores", {})
                if scores and isinstance(scores, dict):
                    result_text += f"\n📊 DETAILED SCORES:\n"
                    result_text += f"  • Combined Score: {scores.get('combined', 'N/A')}/100\n"
                    result_text += f"  • Deep Learning: {scores.get('deep_learning', 'N/A')}/100\n"
                    result_text += f"  • MediaPipe: {scores.get('mediapipe', 'N/A')}/100\n"
            else:
                result_text += f"\n📡 Analysis Method: {analysis_method.upper()}\n"

            result_text += f"\n{'='*50}"

            # Create score visualization HTML
            if score >= 85:
                color = "#28a745"  # Green
                emoji = "🟢"
                label = "EXCELLENT"
            elif score >= 70:
                color = "#ffc107"  # Yellow
                emoji = "🟡"
                label = "GOOD"
            elif score >= 50:
                color = "#fd7e14"  # Orange
                emoji = "🟠"
                label = "FAIR"
            else:
                color = "#dc3545"  # Red
                emoji = "🔴"
                label = "POOR"

            score_html = f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 10px 0;">
                <h1 style="color: white; margin: 0; font-size: 3em;">{emoji}</h1>
                <h2 style="color: white; margin: 10px 0;">Score: {score}/100</h2>
                <h3 style="color: {color}; background: white; display: inline-block; padding: 10px 30px; border-radius: 25px; margin: 10px 0;">{label}</h3>
                <p style="color: white; margin: 15px 0; font-size: 1.1em;">Status: <strong>{status.upper()}</strong></p>
            </div>
            """

            # Annotate image
            annotated_img = np.array(pil_image)

            # Add score overlay
            cv2.putText(
                annotated_img,
                f"Score: {score}/100",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3
            )
            cv2.putText(
                annotated_img,
                f"Status: {status.upper()}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Add colored indicator
            if score >= 85:
                indicator_color = (0, 255, 0)  # Green
            elif score >= 70:
                indicator_color = (0, 255, 255)  # Yellow
            elif score >= 50:
                indicator_color = (0, 165, 255)  # Orange
            else:
                indicator_color = (0, 0, 255)  # Red

            cv2.circle(annotated_img, (30, 30), 20, indicator_color, -1)

            return result_text, annotated_img, score_html

        return "❌ Unexpected response format from agent", None, ""

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Posture Agent.\n\nMake sure it's running:\n   uvicorn main:app --reload --port 8001", None, ""
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Agent took too long to respond.", None, ""
    except Exception as e:
        return f"❌ Error during analysis:\n{str(e)}\n\nDebug info: Check terminal for details", None, ""


def analyze_webcam(image):
    """Analyze posture from webcam capture"""
    if image is None:
        return "❌ No image captured from webcam", None, ""
    return analyze_posture_image(image)


# Create Gradio Interface
with gr.Blocks(
    title="Ergonomic Posture Analyzer",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1400px !important;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    """
) as demo:

    # Header
    gr.HTML("""
    <div class="header">
        <h1>🪑 Ergonomic Posture Analyzer</h1>
        <p style="font-size: 1.1em; margin: 10px 0;">AI-Powered Posture Detection Using MediaPipe & Deep Learning</p>
        <p style="font-size: 0.9em; opacity: 0.9;">Upload an image or use your webcam to analyze your sitting posture</p>
    </div>
    """)

    # Status indicator
    with gr.Row():
        status_box = gr.Textbox(
            label="🔌 Agent Connection Status",
            value="Checking...",
            interactive=False,
            lines=3
        )
        check_status_btn = gr.Button("🔄 Check Status", size="sm")

    def update_status():
        is_running, msg = check_agent_status()
        return msg

    check_status_btn.click(fn=update_status, outputs=status_box)
    demo.load(fn=update_status, outputs=status_box)

    with gr.Tabs():
        # Tab 1: Upload Image
        with gr.Tab("📁 Upload Image"):
            gr.Markdown("""
            ### Upload a Photo
            Upload an image showing your upper body and sitting posture. 
            **Important:** Make sure your shoulders, head, and upper torso are clearly visible from the FRONT (not side view).
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Posture Image",
                        type="pil",
                        height=400
                    )
                    analyze_btn = gr.Button(
                        "🔍 Analyze Posture",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    annotated_output = gr.Image(
                        label="Annotated Result",
                        height=400
                    )

            score_display = gr.HTML(label="Score")
            result_output = gr.Textbox(
                label="Detailed Analysis",
                lines=20,
                max_lines=30
            )

            analyze_btn.click(
                fn=analyze_posture_image,
                inputs=image_input,
                outputs=[result_output, annotated_output, score_display]
            )

        # Tab 2: Webcam
        with gr.Tab("📷 Use Webcam"):
            gr.Markdown("""
            ### Capture from Webcam
            **Setup Instructions:**
            1. Position yourself 2-3 feet from camera
            2. Face the camera DIRECTLY (frontal view, not side)
            3. Ensure upper body is visible
            4. Make sure lighting is good (light in front, not behind)
            5. Click the camera icon to capture
            6. Click "Analyze Webcam Image" button
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        label="Webcam Capture",
                        sources=["webcam"],
                        type="pil",
                        height=400
                    )
                    webcam_analyze_btn = gr.Button(
                        "🔍 Analyze Webcam Image",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    webcam_annotated = gr.Image(
                        label="Annotated Result",
                        height=400
                    )

            webcam_score = gr.HTML(label="Score")
            webcam_result = gr.Textbox(
                label="Detailed Analysis",
                lines=20,
                max_lines=30
            )

            webcam_analyze_btn.click(
                fn=analyze_webcam,
                inputs=webcam_input,
                outputs=[webcam_result, webcam_annotated, webcam_score]
            )

        # Tab 3: How to Use
        with gr.Tab("📖 How to Use"):
            gr.Markdown("""
            ## 🎯 How to Get Accurate Results
            
            ### ✅ CORRECT Setup (Frontal View)
            
            ```
                    📷 Camera
                     ↓
                   👤 YOU
            ========================
            |      CHAIR           |
            ========================
            ```
            
            **Requirements:**
            - ✅ Face camera directly (frontal view)
            - ✅ 2-3 feet distance from camera
            - ✅ Upper body visible (shoulders + head + torso)
            - ✅ Good lighting (in front of you)
            - ✅ Camera at eye level
            
            ### ❌ INCORRECT Setup (Will Give Poor Results)
            
            ```
            ❌ Side View:          ❌ Too Far:           ❌ Backlit:
               📷                     📷                    💡
                ├──→ 👤                ├────→ 👤            ├──→ 👤
            ```
            
            ### 🧪 Testing Different Postures
            
            1. **Good Posture** (Score 85-100):
               - Sit with back straight
               - Shoulders relaxed and level
               - Head aligned with spine
            
            2. **Slouching** (Score 50-70):
               - Round shoulders forward
               - Push head forward
               - Curve back
            
            3. **Severe Slouch** (Score 30-50):
               - Extreme forward head
               - Hunched shoulders
               - Very curved back
            
            ### 🔧 Troubleshooting
            
            | Problem | Solution |
            |---------|----------|
            | "No person detected" | Face camera, improve lighting |
            | Always score 100 | Move closer, ensure upper body visible |
            | Always low score | Check camera angle and distance |
            | Connection error | Start agent: `uvicorn main:app --reload --port 8001` |
            
            ### 📊 Understanding Scores
            
            - **85-100** 🟢 Excellent - Keep it up!
            - **70-84** 🟡 Good - Minor adjustments needed
            - **50-69** 🟠 Fair - Significant improvement needed
            - **0-49** 🔴 Poor - Immediate correction required
            
            ### 🤖 AI Analysis Modes
            
            **Hybrid Mode** (Best - when ML model loaded):
            - Deep Learning Classification (70% weight)
            - MediaPipe Geometric Analysis (30% weight)
            - Confidence scores and probabilities
            
            **MediaPipe-Only Mode** (Fallback):
            - Rule-based geometric analysis
            - Angle and distance measurements
            - Still provides accurate results
            """)

        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 🎯 About This Application
            
            This is an **AI-powered Ergonomic Posture Analyzer** that helps you maintain healthy sitting posture.
            
            ### 🔍 What It Detects
            
            - ✅ **Forward Head Posture** (Tech Neck)
            - ✅ **Slouching** (Rounded Shoulders)
            - ✅ **Spine Alignment** Issues
            - ✅ **Uneven Shoulders**
            - ✅ **Lateral Leaning** (Left/Right)
            
            ### 🛠️ Technology Stack
            
            - **FastAPI**: Backend REST API
            - **TensorFlow/Keras**: Deep Learning (99.05% accuracy)
            - **MediaPipe**: Pose Detection
            - **OpenCV**: Image Processing
            - **Gradio**: Web Interface
            - **Python**: Core Language
            
            ### ⚙️ Requirements
            
            Make sure the Posture Agent is running:
            ```bash
            uvicorn main:app --reload --port 8001
            ```
            
            ### 📊 Agent Information
            
            - **Agent Name**: ergonomic-posture-agent
            - **Version**: 2.0.0
            - **Endpoint**: POST /ergonomic-posture-agent
            - **Health Check**: GET /health
            
            ### 🎓 Model Performance
            
            - **Test Accuracy**: 99.05%
            - **Training Data**: 1,040+ images
            - **Classes**: Good, Bad, Old posture
            - **Architecture**: Transfer Learning (MobileNetV2)
            
            ### 👨‍💻 Created By
            
            University Software Engineering Project - Semester 7
            
            ---
            
            **Version**: 2.0.0  
            **Last Updated**: November 2025  
            **Status**: ✅ Production Ready
            """)

    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e0e0e0;">
        <p style="color: #666; font-size: 0.9em;">
            💡 <strong>Tip:</strong> Maintain good posture: Keep your back straight, shoulders relaxed, 
            and head aligned with your spine. Take breaks every 30 minutes!
        </p>
        <p style="color: #999; font-size: 0.8em; margin-top: 10px;">
            🔗 Make sure agent is running: <code>uvicorn main:app --reload --port 8001</code>
        </p>
    </div>
    """)


if __name__ == "__main__":
    # Check if agent is running before launching
    print("\n" + "="*60)
    print("🚀 Starting Ergonomic Posture Analyzer Web Interface")
    print("="*60)

    is_running, status = check_agent_status()
    print(f"\n{status}")

    if not is_running:
        print("\n⚠️  WARNING: Posture Agent is not running!")
        print("   Start it first with:")
        print("   uvicorn main:app --reload --port 8001\n")
    else:
        print("\n✅ Agent connection successful!")

    print("\n🌐 Launching web interface...")
    print("   Interface will open at: http://127.0.0.1:7860")
    print("="*60 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )