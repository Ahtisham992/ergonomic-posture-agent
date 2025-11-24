import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, RefreshCw, AlertCircle, CheckCircle, Activity, X } from 'lucide-react';

// Your Hugging Face Space URL
const AGENT_URL = "https://ahtisham992-ergonomic-posture-agent.hf.space/ergonomic-posture-agent";
const HEALTH_URL = "https://ahtisham992-ergonomic-posture-agent.hf.space/health";

const PostureAnalyzer = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [agentStatus, setAgentStatus] = useState({ ready: false, message: 'Checking...' });
  const [uploadedImage, setUploadedImage] = useState(null);
  const [webcamImage, setWebcamImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stream, setStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Check agent status on mount
  useEffect(() => {
    checkAgentStatus();
  }, []);

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const checkAgentStatus = async () => {
    try {
      const response = await fetch(HEALTH_URL);
      const data = await response.json();
      
      if (data.ready) {
        setAgentStatus({
          ready: true,
          message: `✅ Agent is running: ${data.agent_name}\n   Mode: ${data.analysis_mode?.toUpperCase()}\n   ${data.ml_model_loaded ? '🤖 ML Model: Loaded' : '⚠️ ML Model: Not loaded (MediaPipe only)'}`
        });
      } else {
        setAgentStatus({ ready: false, message: '⚠️ Agent not ready' });
      }
    } catch (err) {
      setAgentStatus({ ready: false, message: '❌ Cannot connect to agent' });
    }
  };

  const imageToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const base64 = await imageToBase64(file);
      setUploadedImage(base64);
      setResult(null);
      setError(null);
    }
  };

  const startWebcam = async () => {
    try {
      setError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      setStream(mediaStream);
      setIsCameraActive(true);
      
      // Wait a bit for video element to be ready
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.play().catch(err => {
            console.error('Error playing video:', err);
            setError('Failed to start video playback');
          });
        }
      }, 100);
    } catch (err) {
      console.error('Camera error:', err);
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access in your browser settings.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera and try again.');
      } else {
        setError(`Unable to access camera: ${err.message}`);
      }
      setIsCameraActive(false);
    }
  };

  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsCameraActive(false);
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) {
      setError('Video or canvas not ready');
      return;
    }
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Check if video is actually playing
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      setError('Video is not ready yet. Please wait a moment and try again.');
      return;
    }
    
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    
    // Draw the video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.95);
    setWebcamImage(imageData);
    stopWebcam();
    setResult(null);
    setError(null);
  };

  const analyzePosture = async (imageData) => {
    if (!imageData) {
      setError('Please upload or capture an image first');
      return;
    }

    if (!agentStatus.ready) {
      setError('Agent is not ready. Please check the connection.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const requestData = {
        messages: [
          {
            role: "user",
            content: imageData
          }
        ]
      };

      const response = await fetch(AGENT_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      const data = await response.json();
      
      // Debug logging
      console.log('Full API Response:', JSON.stringify(data, null, 2));
      console.log('Posture Analysis:', data.data?.posture_analysis);

      if (data.status === 'error') {
        setError(data.error_message || 'Analysis failed');
        return;
      }

      if (data.status === 'success' && data.data) {
        const analysis = data.data.posture_analysis;
        
        // Extract metrics from mediapipe_analysis if not in top level
        let metrics = analysis.metrics || {};
        let issues = analysis.issues || [];
        
        // Check if metrics are nested in mediapipe_analysis
        if (analysis.mediapipe_analysis) {
          const mpAnalysis = analysis.mediapipe_analysis;
          if (mpAnalysis.metrics && Object.keys(metrics).length === 0) {
            metrics = mpAnalysis.metrics;
          }
          if (mpAnalysis.issues && issues.length === 0) {
            issues = mpAnalysis.issues;
          }
        }
        
        console.log('Parsed metrics:', metrics);
        console.log('Parsed issues:', issues);
        
        setResult({
          score: analysis.posture_score,
          status: analysis.posture_status,
          feedback: analysis.feedback || data.data.message,
          metrics: metrics,
          issues: issues,
          method: analysis.analysis_method,
          dlClassification: analysis.dl_classification,
          scores: analysis.scores,
          mediapipeAnalysis: analysis.mediapipe_analysis
        });
      }
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 85) return { bg: '#28a745', label: 'EXCELLENT', emoji: '🟢' };
    if (score >= 70) return { bg: '#ffc107', label: 'GOOD', emoji: '🟡' };
    if (score >= 50) return { bg: '#fd7e14', label: 'FAIR', emoji: '🟠' };
    return { bg: '#dc3545', label: 'POOR', emoji: '🔴' };
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '10px 15px' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ 
          background: 'white', 
          borderRadius: '15px', 
          padding: '20px 15px', 
          marginBottom: '15px',
          textAlign: 'center',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <h1 style={{ margin: '0 0 10px 0', color: '#667eea', fontSize: 'clamp(1.5em, 5vw, 2.5em)' }}>
            🪑 Ergonomic Posture Analyzer
          </h1>
          <p style={{ color: '#666', fontSize: 'clamp(0.9em, 2.5vw, 1.1em)', margin: '10px 0' }}>
            AI-Powered Posture Detection Using MediaPipe & Deep Learning
          </p>
          <p style={{ color: '#999', fontSize: 'clamp(0.8em, 2vw, 0.9em)' }}>
            Upload an image or use your webcam to analyze your sitting posture
          </p>
        </div>

        {/* Agent Status */}
        <div style={{ 
          background: 'white', 
          borderRadius: '15px', 
          padding: '15px', 
          marginBottom: '15px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', flexDirection: window.innerWidth < 768 ? 'column' : 'row', gap: '15px', justifyContent: 'space-between', alignItems: window.innerWidth < 768 ? 'stretch' : 'center' }}>
            <div style={{ flex: 1 }}>
              <h3 style={{ margin: '0 0 10px 0', color: '#333', fontSize: 'clamp(1em, 3vw, 1.2em)' }}>📌 Agent Connection Status</h3>
              <pre style={{ 
                background: '#f5f5f5', 
                padding: '10px', 
                borderRadius: '5px',
                margin: 0,
                fontSize: 'clamp(0.75em, 2vw, 0.9em)',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                overflowX: 'auto'
              }}>
                {agentStatus.message}
              </pre>
            </div>
            <button
              onClick={checkAgentStatus}
              style={{
                padding: '12px 20px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                fontSize: 'clamp(0.9em, 2.5vw, 1em)',
                whiteSpace: 'nowrap',
                alignSelf: window.innerWidth < 768 ? 'stretch' : 'center'
              }}
            >
              <RefreshCw size={18} /> Check Status
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ 
          background: 'white', 
          borderRadius: '15px', 
          overflow: 'hidden',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', borderBottom: '2px solid #eee' }}>
            <button
              onClick={() => setActiveTab('upload')}
              style={{
                flex: 1,
                padding: '20px',
                background: activeTab === 'upload' ? '#667eea' : 'white',
                color: activeTab === 'upload' ? 'white' : '#666',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1.1em',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px'
              }}
            >
              <Upload size={20} /> Upload Image
            </button>
            <button
              onClick={() => setActiveTab('webcam')}
              style={{
                flex: 1,
                padding: '20px',
                background: activeTab === 'webcam' ? '#667eea' : 'white',
                color: activeTab === 'webcam' ? 'white' : '#666',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1.1em',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px'
              }}
            >
              <Camera size={20} /> Use Webcam
            </button>
            <button
              onClick={() => setActiveTab('guide')}
              style={{
                flex: 1,
                padding: '20px',
                background: activeTab === 'guide' ? '#667eea' : 'white',
                color: activeTab === 'guide' ? 'white' : '#666',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1.1em',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px'
              }}
            >
              <Activity size={20} /> How to Use
            </button>
          </div>

          <div style={{ padding: '20px 15px' }}>
            {/* Upload Tab */}
            {activeTab === 'upload' && (
              <div>
                <h3 style={{ color: '#333', marginBottom: '15px', fontSize: 'clamp(1.1em, 3vw, 1.3em)' }}>📤 Upload a Photo</h3>
                <p style={{ color: '#666', marginBottom: '20px', fontSize: 'clamp(0.9em, 2.5vw, 1em)' }}>
                  Upload an image showing your upper body and sitting posture. Make sure your shoulders, head, and upper torso are clearly visible from the FRONT.
                </p>
                
                <div style={{ display: 'grid', gridTemplateColumns: window.innerWidth < 768 ? '1fr' : '1fr 1fr', gap: '20px' }}>
                  <div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      style={{ display: 'none' }}
                    />
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      style={{
                        border: '2px dashed #667eea',
                        borderRadius: '10px',
                        padding: '30px 20px',
                        textAlign: 'center',
                        cursor: 'pointer',
                        background: uploadedImage ? '#f0f0f0' : 'white',
                        minHeight: window.innerWidth < 768 ? '250px' : '400px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      {uploadedImage ? (
                        <img src={uploadedImage} alt="Uploaded" style={{ maxWidth: '100%', maxHeight: window.innerWidth < 768 ? '200px' : '350px', borderRadius: '8px' }} />
                      ) : (
                        <>
                          <Upload size={window.innerWidth < 768 ? 36 : 48} color="#667eea" />
                          <p style={{ marginTop: '20px', color: '#666', fontSize: 'clamp(0.9em, 2.5vw, 1em)' }}>Click to upload image</p>
                        </>
                      )}
                    </div>
                    <button
                      onClick={() => analyzePosture(uploadedImage)}
                      disabled={!uploadedImage || loading}
                      style={{
                        width: '100%',
                        marginTop: '15px',
                        padding: '15px',
                        background: uploadedImage && !loading ? '#667eea' : '#ccc',
                        color: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        cursor: uploadedImage && !loading ? 'pointer' : 'not-allowed',
                        fontSize: 'clamp(1em, 2.5vw, 1.1em)',
                        fontWeight: 'bold'
                      }}
                    >
                      {loading ? '🔄 Analyzing...' : '🔍 Analyze Posture'}
                    </button>
                  </div>
                  
                  <div>
                    {result && <ResultDisplay result={result} />}
                    {error && <ErrorDisplay message={error} />}
                  </div>
                </div>
              </div>
            )}

            {/* Webcam Tab */}
            {activeTab === 'webcam' && (
              <div>
                <h3 style={{ color: '#333', marginBottom: '15px', fontSize: 'clamp(1.1em, 3vw, 1.3em)' }}>📷 Capture from Webcam</h3>
                <p style={{ color: '#666', marginBottom: '20px', fontSize: 'clamp(0.9em, 2.5vw, 1em)' }}>
                  Position yourself 2-3 feet from camera. Face the camera directly. Ensure upper body is visible.
                </p>
                
                <div style={{ display: 'grid', gridTemplateColumns: window.innerWidth < 768 ? '1fr' : '1fr 1fr', gap: '20px' }}>
                  <div>
                    {!webcamImage && (
                      <div style={{
                        border: '2px solid #667eea',
                        borderRadius: '10px',
                        overflow: 'hidden',
                        background: '#000',
                        minHeight: window.innerWidth < 768 ? '300px' : '480px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        position: 'relative'
                      }}>
                        {isCameraActive ? (
                          <>
                            <video
                              ref={videoRef}
                              autoPlay
                              playsInline
                              muted
                              style={{ 
                                width: '100%', 
                                height: 'auto',
                                maxHeight: window.innerWidth < 768 ? '300px' : '480px',
                                objectFit: 'cover'
                              }}
                            />
                            <div style={{
                              position: 'absolute',
                              top: '10px',
                              left: '10px',
                              background: 'rgba(0, 200, 0, 0.8)',
                              color: 'white',
                              padding: '5px 10px',
                              borderRadius: '5px',
                              fontSize: 'clamp(0.8em, 2vw, 0.9em)'
                            }}>
                              ● LIVE
                            </div>
                          </>
                        ) : (
                          <div style={{ textAlign: 'center', color: 'white', padding: '20px' }}>
                            <Camera size={window.innerWidth < 768 ? 36 : 48} />
                            <p style={{ marginTop: '20px', fontSize: 'clamp(0.9em, 2.5vw, 1em)' }}>Click "Open Webcam" to start</p>
                            <p style={{ fontSize: 'clamp(0.8em, 2vw, 0.9em)', opacity: 0.7 }}>Make sure to allow camera permissions</p>
                          </div>
                        )}
                      </div>
                    )}
                    {webcamImage && (
                      <div style={{ position: 'relative' }}>
                        <img src={webcamImage} alt="Captured" style={{ width: '100%', borderRadius: '10px' }} />
                        <button
                          onClick={() => setWebcamImage(null)}
                          style={{
                            position: 'absolute',
                            top: '10px',
                            right: '10px',
                            background: 'rgba(220, 53, 69, 0.9)',
                            color: 'white',
                            border: 'none',
                            borderRadius: '50%',
                            width: '40px',
                            height: '40px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}
                        >
                          <X size={20} />
                        </button>
                      </div>
                    )}
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                    
                    <div style={{ display: 'flex', gap: '10px', marginTop: '15px', flexWrap: 'wrap' }}>
                      {!isCameraActive && !webcamImage && (
                        <button
                          onClick={startWebcam}
                          style={{
                            flex: 1,
                            minWidth: '150px',
                            padding: '15px',
                            background: '#28a745',
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: 'clamp(1em, 2.5vw, 1.1em)',
                            fontWeight: 'bold'
                          }}
                        >
                          📹 Open Webcam
                        </button>
                      )}
                      {isCameraActive && (
                        <>
                          <button
                            onClick={captureImage}
                            style={{
                              flex: 1,
                              minWidth: '120px',
                              padding: '15px',
                              background: '#667eea',
                              color: 'white',
                              border: 'none',
                              borderRadius: '8px',
                              cursor: 'pointer',
                              fontSize: 'clamp(1em, 2.5vw, 1.1em)',
                              fontWeight: 'bold'
                            }}
                          >
                            📸 Capture Photo
                          </button>
                          <button
                            onClick={stopWebcam}
                            style={{
                              padding: '15px 25px',
                              background: '#dc3545',
                              color: 'white',
                              border: 'none',
                              borderRadius: '8px',
                              cursor: 'pointer',
                              fontSize: 'clamp(1em, 2.5vw, 1.1em)',
                              fontWeight: 'bold'
                            }}
                          >
                            ❌
                          </button>
                        </>
                      )}
                      {webcamImage && (
                        <button
                          onClick={() => analyzePosture(webcamImage)}
                          disabled={loading}
                          style={{
                            flex: 1,
                            minWidth: '150px',
                            padding: '15px',
                            background: loading ? '#ccc' : '#667eea',
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: loading ? 'not-allowed' : 'pointer',
                            fontSize: 'clamp(1em, 2.5vw, 1.1em)',
                            fontWeight: 'bold'
                          }}
                        >
                          {loading ? '🔄 Analyzing...' : '🔍 Analyze Posture'}
                        </button>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    {result && <ResultDisplay result={result} />}
                    {error && <ErrorDisplay message={error} />}
                  </div>
                </div>
              </div>
            )}

            {/* Guide Tab */}
            {activeTab === 'guide' && (
              <div style={{ color: '#333', fontSize: 'clamp(0.9em, 2.5vw, 1em)' }}>
                <h2 style={{ fontSize: 'clamp(1.3em, 4vw, 1.8em)' }}>🎯 How to Get Accurate Results</h2>
                
                <h3 style={{ marginTop: '30px', color: '#667eea', fontSize: 'clamp(1.1em, 3vw, 1.3em)' }}>✅ CORRECT Setup (Frontal View)</h3>
                <div style={{ background: '#f5f5f5', padding: '15px', borderRadius: '10px', marginTop: '10px', overflowX: 'auto' }}>
                  <pre style={{ margin: 0, fontSize: 'clamp(0.7em, 2vw, 0.9em)' }}>{`
        📷 Camera
         ↓
       👤 YOU
========================
|      CHAIR           |
========================
                  `}</pre>
                </div>
                
                <h3 style={{ marginTop: '30px', color: '#667eea', fontSize: 'clamp(1.1em, 3vw, 1.3em)' }}>Requirements:</h3>
                <ul style={{ lineHeight: '2', paddingLeft: '20px' }}>
                  <li>✅ Face camera directly (frontal view)</li>
                  <li>✅ 2-3 feet distance from camera</li>
                  <li>✅ Upper body visible (shoulders + head + torso)</li>
                  <li>✅ Good lighting (in front of you)</li>
                  <li>✅ Camera at eye level</li>
                </ul>
                
                <h3 style={{ marginTop: '30px', color: '#667eea', fontSize: 'clamp(1.1em, 3vw, 1.3em)' }}>📊 Understanding Scores</h3>
                <ul style={{ lineHeight: '2', paddingLeft: '20px' }}>
                  <li><strong>85-100 🟢 Excellent</strong> - Keep it up!</li>
                  <li><strong>70-84 🟡 Good</strong> - Minor adjustments needed</li>
                  <li><strong>50-69 🟠 Fair</strong> - Significant improvement needed</li>
                  <li><strong>0-49 🔴 Poor</strong> - Immediate correction required</li>
                </ul>

                <h3 style={{ marginTop: '30px', color: '#667eea', fontSize: 'clamp(1.1em, 3vw, 1.3em)' }}>🤖 AI Analysis Modes</h3>
                <p><strong>Hybrid Mode</strong> (Best):</p>
                <ul style={{ paddingLeft: '20px' }}>
                  <li>Deep Learning Classification (70% weight)</li>
                  <li>MediaPipe Geometric Analysis (30% weight)</li>
                  <li>Confidence scores and probabilities</li>
                </ul>
                
                <p style={{ marginTop: '15px' }}><strong>MediaPipe-Only Mode</strong> (Fallback):</p>
                <ul style={{ paddingLeft: '20px' }}>
                  <li>Rule-based geometric analysis</li>
                  <li>Still accurate for most cases</li>
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{ 
          background: 'white', 
          borderRadius: '15px', 
          padding: '15px',
          marginTop: '15px',
          textAlign: 'center',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <p style={{ color: '#666', margin: '10px 0', fontSize: 'clamp(0.85em, 2vw, 1em)', lineHeight: '1.6' }}>
            💡 <strong>Tip:</strong> Maintain good posture: Keep your back straight, shoulders relaxed, 
            and head aligned with your spine. Take breaks every 30 minutes!
          </p>
          <p style={{ color: '#999', fontSize: 'clamp(0.75em, 1.8vw, 0.9em)', marginTop: '10px' }}>
            Powered by MediaPipe & Deep Learning | Version 2.1.0
          </p>
        </div>
      </div>
    </div>
  );
};

const ResultDisplay = ({ result }) => {
  const scoreInfo = result.score >= 85 ? { bg: '#28a745', label: 'EXCELLENT', emoji: '🟢' } :
                    result.score >= 70 ? { bg: '#ffc107', label: 'GOOD', emoji: '🟡' } :
                    result.score >= 50 ? { bg: '#fd7e14', label: 'FAIR', emoji: '🟠' } :
                    { bg: '#dc3545', label: 'POOR', emoji: '🔴' };

  return (
    <div>
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '15px',
        padding: '20px 15px',
        textAlign: 'center',
        color: 'white',
        marginBottom: '20px'
      }}>
        <div style={{ fontSize: 'clamp(2em, 8vw, 3em)', margin: '0' }}>{scoreInfo.emoji}</div>
        <h2 style={{ margin: '10px 0', fontSize: 'clamp(1.3em, 5vw, 2em)' }}>Score: {result.score}/100</h2>
        <div style={{
          background: 'white',
          color: scoreInfo.bg,
          display: 'inline-block',
          padding: '8px 25px',
          borderRadius: '25px',
          margin: '10px 0',
          fontWeight: 'bold',
          fontSize: 'clamp(1em, 3vw, 1.2em)'
        }}>
          {scoreInfo.label}
        </div>
        <p style={{ margin: '15px 0', fontSize: 'clamp(0.95em, 2.5vw, 1.1em)' }}>Status: <strong>{result.status.toUpperCase()}</strong></p>
      </div>

      <div style={{ 
        background: '#f8f9fa', 
        borderRadius: '10px', 
        padding: '15px',
        maxHeight: '600px',
        overflowY: 'auto',
        fontSize: 'clamp(0.85em, 2vw, 1em)'
      }}>
        <div style={{ 
          background: 'white', 
          padding: '12px', 
          borderRadius: '8px',
          marginBottom: '15px',
          border: '2px solid #667eea'
        }}>
          <h3 style={{ color: '#667eea', marginTop: 0, marginBottom: '10px', fontSize: 'clamp(1em, 3vw, 1.2em)' }}>🎯 POSTURE ANALYSIS RESULTS</h3>
          <div style={{ color: '#333', fontSize: '0.95em', lineHeight: '1.8' }}>
            <p><strong>📊 POSTURE SCORE:</strong> {result.score}/100</p>
            <p><strong>🔍 STATUS:</strong> {result.status.toUpperCase()}</p>
          </div>
        </div>

        <h3 style={{ color: '#333', marginTop: '20px', marginBottom: '10px', fontSize: 'clamp(1em, 2.8vw, 1.15em)' }}>💬 FEEDBACK</h3>
        <p style={{ color: '#666', lineHeight: '1.6', background: 'white', padding: '12px', borderRadius: '8px' }}>
          {result.feedback}
        </p>

        <h3 style={{ color: '#333', marginTop: '20px', marginBottom: '10px', fontSize: 'clamp(1em, 2.8vw, 1.15em)' }}>🔬 DETAILED METRICS</h3>
        <div style={{ background: 'white', padding: '12px', borderRadius: '8px' }}>
          <ul style={{ color: '#666', lineHeight: '2', margin: 0, paddingLeft: '20px' }}>
            <li><strong>Spine Angle:</strong> {result.metrics.spine_angle ? `${result.metrics.spine_angle}°` : 'N/A'}</li>
            <li><strong>Shoulder Slope:</strong> {result.metrics.shoulder_slope ? result.metrics.shoulder_slope : 'N/A'}</li>
            <li><strong>Head Forward Distance:</strong> {result.metrics.head_forward_distance ? result.metrics.head_forward_distance : 'N/A'}</li>
            {result.metrics.head_shoulder_vertical && (
              <li><strong>Head-Shoulder Vertical:</strong> {result.metrics.head_shoulder_vertical}</li>
            )}
          </ul>
        </div>

        {result.issues && result.issues.length > 0 ? (
          <>
            <h3 style={{ color: '#dc3545', marginTop: '20px', marginBottom: '10px', fontSize: 'clamp(1em, 2.8vw, 1.15em)' }}>⚠️ ISSUES DETECTED</h3>
            <div style={{ background: 'white', padding: '12px', borderRadius: '8px' }}>
              <ul style={{ color: '#666', lineHeight: '2', margin: 0, paddingLeft: '20px' }}>
                {result.issues.map((issue, i) => (
                  <li key={i}>{issue}</li>
                ))}
              </ul>
            </div>
          </>
        ) : (
          <div style={{ 
            background: '#d4edda', 
            padding: '12px', 
            borderRadius: '8px',
            marginTop: '20px',
            color: '#155724'
          }}>
            <p style={{ margin: 0 }}>✅ <strong>NO MAJOR ISSUES DETECTED!</strong></p>
          </div>
        )}

        {result.dlClassification && (
          <>
            <h3 style={{ color: '#333', marginTop: '20px', marginBottom: '10px', fontSize: 'clamp(1em, 2.8vw, 1.15em)' }}>🤖 AI CLASSIFICATION</h3>
            <div style={{ background: 'white', padding: '12px', borderRadius: '8px' }}>
              <ul style={{ color: '#666', lineHeight: '2', margin: 0, paddingLeft: '20px' }}>
                <li><strong>Predicted Class:</strong> {result.dlClassification.predicted_class.toUpperCase()}</li>
                <li><strong>Confidence:</strong> {(result.dlClassification.confidence * 100).toFixed(1)}%</li>
              </ul>
              
              {result.dlClassification.all_probabilities && (
                <div style={{ marginTop: '15px' }}>
                  <p style={{ margin: '0 0 10px 0', color: '#666', fontWeight: 'bold' }}>All Probabilities:</p>
                  <ul style={{ color: '#666', lineHeight: '1.8', margin: 0, paddingLeft: '40px' }}>
                    {Object.entries(result.dlClassification.all_probabilities).map(([cls, prob]) => (
                      <li key={cls}>
                        <strong>{cls}:</strong> {(prob * 100).toFixed(1)}%
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </>
        )}

        {result.scores && (
          <>
            <h3 style={{ color: '#333', marginTop: '20px', marginBottom: '10px', fontSize: 'clamp(1em, 2.8vw, 1.15em)' }}>📊 DETAILED SCORES</h3>
            <div style={{ background: 'white', padding: '12px', borderRadius: '8px' }}>
              <ul style={{ color: '#666', lineHeight: '2', margin: 0, paddingLeft: '20px' }}>
                <li><strong>Combined Score:</strong> {result.scores.combined}/100</li>
                <li><strong>Deep Learning:</strong> {result.scores.deep_learning}/100</li>
                <li><strong>MediaPipe:</strong> {result.scores.mediapipe}/100</li>
              </ul>
            </div>
          </>
        )}

        {result.method && (
          <div style={{ 
            marginTop: '20px', 
            padding: '10px 12px', 
            background: '#e7f3ff',
            borderRadius: '8px',
            color: '#004085',
            fontSize: 'clamp(0.8em, 2vw, 0.9em)'
          }}>
            <strong>📡 Analysis Method:</strong> {result.method.toUpperCase().replace('_', ' ')}
          </div>
        )}
      </div>
    </div>
  );
};

const ErrorDisplay = ({ message }) => (
  <div style={{
    background: '#f8d7da',
    border: '1px solid #f5c6cb',
    borderRadius: '10px',
    padding: '20px',
    color: '#721c24'
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
      <AlertCircle size={24} />
      <h3 style={{ margin: 0 }}>Error</h3>
    </div>
    <p style={{ margin: 0 }}>{message}</p>
  </div>
);

export default PostureAnalyzer;