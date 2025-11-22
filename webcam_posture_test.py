"""
Webcam Posture Tester
Captures image from webcam and sends to Ergonomic Posture Agent for analysis
"""

import cv2
import requests
import json
import time
from datetime import datetime

# Configuration
AGENT_URL = "http://localhost:8001/posture-agent"
CAMERA_INDEX = 0  # 0 for default webcam, change if you have multiple cameras
COUNTDOWN_SECONDS = 3


def capture_from_webcam():
    """
    Capture image from webcam with countdown

    Returns:
        tuple: (success, image_array or error_message)
    """
    print("🎥 Initializing webcam...")

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        return False, "Failed to open webcam. Check if camera is connected."

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ Webcam initialized")
    print("\n📸 Position yourself in front of the camera")
    print("   - Ensure upper body is visible")
    print("   - Face the camera directly")
    print("   - Good lighting is important\n")

    # Countdown
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        ret, frame = cap.read()
        if ret:
            # Display countdown on frame
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"Capturing in {i}...",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3
            )
            cv2.putText(
                display_frame,
                "Position yourself for posture analysis",
                (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.imshow('Webcam - Posture Capture', display_frame)
            cv2.waitKey(1000)

        print(f"⏱️  {i}...")

    # Capture final image
    ret, frame = cap.read()

    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        return False, "Failed to capture image from webcam"

    # Show captured image
    print("📸 Image captured!")
    cv2.putText(
        frame,
        "CAPTURED!",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )
    cv2.imshow('Webcam - Posture Capture', frame)
    cv2.waitKey(2000)

    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"posture_capture_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"💾 Image saved as: {filename}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    return True, frame


def analyze_posture(image):
    """
    Send captured image to posture agent for analysis

    Args:
        image: OpenCV image array (numpy array)

    Returns:
        dict: Analysis results
    """
    print("\n🤖 Sending image to Posture Agent...")

    try:
        # Encode image as JPEG
        success, encoded_image = cv2.imencode('.jpg', image)

        if not success:
            return {"error": "Failed to encode image"}

        # Prepare multipart form data
        files = {
            'file': ('posture.jpg', encoded_image.tobytes(), 'image/jpeg')
        }

        data = {
            'request': json.dumps({
                "messages": [
                    {"role": "user", "content": "Analyze my posture from webcam"}
                ]
            })
        }

        # Send POST request
        response = requests.post(
            AGENT_URL,
            files=files,
            data=data,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Request failed with status code {response.status_code}",
                "details": response.text
            }

    except requests.exceptions.ConnectionError:
        return {
            "error": "Cannot connect to Posture Agent. Make sure it's running on port 8001"
        }
    except Exception as e:
        return {
            "error": f"Error during analysis: {str(e)}"
        }


def display_results(result):
    """
    Display analysis results in a formatted way

    Args:
        result: Response from posture agent
    """
    print("\n" + "=" * 60)
    print("📊 POSTURE ANALYSIS RESULTS")
    print("=" * 60)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "details" in result:
            print(f"   Details: {result['details']}")
        return

    if result.get("status") == "error":
        print(f"❌ Error: {result.get('error_message', 'Unknown error')}")
        return

    if result.get("status") == "success":
        data = result.get("data", {})

        # Posture Score with color coding
        score = data.get("posture_score", 0)
        if score >= 85:
            score_emoji = "🟢"
            score_label = "EXCELLENT"
        elif score >= 70:
            score_emoji = "🟡"
            score_label = "GOOD"
        elif score >= 50:
            score_emoji = "🟠"
            score_label = "FAIR"
        else:
            score_emoji = "🔴"
            score_label = "POOR"

        print(f"\n{score_emoji} Posture Score: {score}/100 - {score_label}")
        print(f"📍 Status: {data.get('posture_status', 'N/A').upper()}")
        print(f"\n💬 Feedback:")
        print(f"   {data.get('feedback', 'No feedback available')}")

        # Detailed metrics
        metrics = data.get("metrics", {})
        if metrics:
            print(f"\n📐 Detailed Metrics:")
            print(f"   • Spine Angle: {metrics.get('spine_angle', 'N/A')}°")
            print(f"   • Shoulder Slope: {metrics.get('shoulder_slope', 'N/A')}")
            print(f"   • Head Forward Distance: {metrics.get('head_forward_distance', 'N/A')}")

            issues = metrics.get('issues', [])
            if issues:
                print(f"\n⚠️  Issues Detected:")
                for issue in issues:
                    print(f"   • {issue}")
            else:
                print(f"\n✅ No issues detected!")

    print("=" * 60)


def main():
    """
    Main function to run webcam capture and posture analysis
    """
    print("\n" + "=" * 60)
    print("🎯 WEBCAM POSTURE ANALYZER")
    print("=" * 60)
    print("This tool will:")
    print("  1. Capture your image from webcam")
    print("  2. Send it to the Posture Agent")
    print("  3. Display posture analysis results")
    print("=" * 60 + "\n")

    # Check if agent is running
    print("🔍 Checking if Posture Agent is running...")
    try:
        health_check = requests.get("http://localhost:8001/health", timeout=5)
        if health_check.status_code == 200:
            print("✅ Posture Agent is running\n")
        else:
            print("⚠️  Posture Agent responded but with unexpected status\n")
    except:
        print("❌ ERROR: Posture Agent is not running!")
        print("   Please start it first with: uvicorn main:app --reload --port 8001\n")
        return

    # Capture from webcam
    success, result = capture_from_webcam()

    if not success:
        print(f"❌ Error: {result}")
        return

    # Analyze posture
    analysis_result = analyze_posture(result)

    # Display results
    display_results(analysis_result)

    # Ask if user wants to analyze another image
    print("\n" + "=" * 60)
    response = input("📸 Capture and analyze another image? (y/n): ").strip().lower()

    if response == 'y':
        print("\n")
        main()  # Recursive call for another capture
    else:
        print("\n👋 Thanks for using Webcam Posture Analyzer!")
        print("   Keep maintaining good posture! 🪑\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting... Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("   Please try again or check the error logs.")