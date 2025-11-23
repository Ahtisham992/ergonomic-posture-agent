"""
Diagnose Connection Issue Between Web Interface and Agent
Run this to see what's going wrong
"""

import requests
import time
import json

print("🔍 DIAGNOSING CONNECTION TO AGENT")
print("=" * 80)

AGENT_URL = "http://localhost:8001"
HEALTH_URL = f"{AGENT_URL}/health"
ROOT_URL = f"{AGENT_URL}/"

# Test 1: Check if port 8001 is accessible
print("\n1️⃣ Testing if port 8001 is accessible...")
try:
    response = requests.get(ROOT_URL, timeout=5)
    print(f"   ✅ Port 8001 is accessible")
    print(f"   Response: {response.status_code}")
    print(f"   Data: {response.json()}")
except requests.exceptions.ConnectionError as e:
    print(f"   ❌ Cannot connect to port 8001")
    print(f"   Error: {e}")
    print("\n   SOLUTION:")
    print("   1. Make sure the agent is running:")
    print("      uvicorn main:app --reload --port 8001")
    print("   2. Check if another program is using port 8001")
    exit(1)
except Exception as e:
    print(f"   ⚠️ Unexpected error: {e}")

# Test 2: Check health endpoint
print("\n2️⃣ Testing /health endpoint...")
for attempt in range(3):
    try:
        print(f"   Attempt {attempt + 1}/3...")
        response = requests.get(HEALTH_URL, timeout=30)
        print(f"   ✅ Health check successful!")
        print(f"   Status code: {response.status_code}")

        data = response.json()
        print(f"   Agent ready: {data.get('ready', False)}")
        print(f"   ML model loaded: {data.get('ml_model_loaded', False)}")
        print(f"   Analysis mode: {data.get('analysis_mode', 'unknown')}")
        break
    except requests.exceptions.Timeout:
        print(f"   ⏱️ Timeout on attempt {attempt + 1}")
        if attempt < 2:
            print(f"   Waiting 5 seconds before retry...")
            time.sleep(5)
        else:
            print("\n   ❌ Health check timed out after 3 attempts")
            print("\n   POSSIBLE CAUSES:")
            print("   1. TensorFlow is still initializing (takes 30-60 seconds)")
            print("   2. Agent crashed during startup")
            print("   3. Server is overloaded")
            print("\n   SOLUTION:")
            print("   Check the server terminal for errors")
            print("   Look for: 'Server ready to accept requests'")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        break

# Test 3: Check if agent endpoint exists
print("\n3️⃣ Testing /ergonomic-posture-agent endpoint...")
try:
    # Try a simple GET request (should return 405 Method Not Allowed)
    response = requests.get(f"{AGENT_URL}/ergonomic-posture-agent", timeout=5)
    if response.status_code == 405:
        print(f"   ✅ Endpoint exists (returns 405 as expected for GET)")
    else:
        print(f"   ⚠️ Unexpected status: {response.status_code}")
except requests.exceptions.Timeout:
    print(f"   ⏱️ Timeout - endpoint is taking too long")
except Exception as e:
    print(f"   ℹ️ Response: {e}")

# Test 4: Check server logs suggest
print("\n4️⃣ Checking what the server should show...")
print("   Your server terminal should display:")
print("   ✅ 'Loading trained deep learning model...'")
print("   ✅ 'Architecture built'")
print("   ✅ 'Weights loaded'")
print("   ✅ 'Model compiled'")
print("   ✅ 'Model test successful!'")
print("   ✅ 'Deep learning model loaded successfully!'")
print("   ✅ 'Running in HYBRID mode'")
print("   ✅ 'Server ready to accept requests'")
print("   ✅ 'Application startup complete.'")

# Test 5: Performance test
print("\n5️⃣ Testing response time...")
try:
    start = time.time()
    response = requests.get(ROOT_URL, timeout=10)
    duration = time.time() - start
    print(f"   ⏱️ Response time: {duration:.2f} seconds")

    if duration > 5:
        print(f"   ⚠️ Server is slow (>{duration:.2f}s)")
        print("   This might cause timeout issues")
    else:
        print(f"   ✅ Response time is good")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Summary
print("\n" + "=" * 80)
print("📋 DIAGNOSIS SUMMARY")
print("=" * 80)

print("\nIf you see errors above, follow these steps:")
print("\n1. RESTART THE AGENT SERVER:")
print("   - Press Ctrl+C in the server terminal")
print("   - Run: uvicorn main:app --reload --port 8001")
print("   - Wait for 'Application startup complete.'")
print("")
print("2. WAIT FOR TENSORFLOW TO INITIALIZE:")
print("   - First startup takes 30-60 seconds")
print("   - Watch for 'Model test successful!' message")
print("")
print("3. THEN START WEB INTERFACE:")
print("   - python web_interface.py")
print("")
print("4. IF STILL FAILING:")
print("   - Check if firewall is blocking port 8001")
print("   - Check if antivirus is interfering")
print("   - Try changing to port 8002 in both files")

print("\n" + "=" * 80)