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
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsCameraActive(true);
      setError(null);
    } catch (err) {
      setError('Unable to access camera. Please check permissions.');
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
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
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

      if (data.status === 'error') {
        setError(data.error_message || 'Analysis failed');
        return;
      }

      if (data.status === 'success' && data.data) {
        const analysis = data.data.posture_analysis;
        setResult({
          score: analysis.posture_score,
          status: analysis.posture_status,
          feedback: analysis.feedback || data.data.message,
          metrics: analysis.metrics || {},
          issues: analysis.issues || [],
          method: analysis.analysis_method,
          dlClassification: analysis.dl_classification,
          scores: analysis.scores
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
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '20px' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ 
          background: 'white', 
          borderRadius: '15px', 
          padding: '30px', 
          marginBottom: '20px',
          textAlign: 'center',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <h1 style={{ margin: '0 0 10px 0', color: '#667eea', fontSize: '2.5em' }}>
            🪑 Ergonomic Posture Analyzer
          </h1>
          <p style={{ color: '#666', fontSize: '1.1em', margin: '10px 0' }}>
            AI-Powered Posture Detection Using MediaPipe & Deep Learning
          </p>
          <p style={{ color: '#999', fontSize: '0.9em' }}>
            Upload an image or use your webcam to analyze your sitting posture
          </p>
        </div>

        {/* Agent Status */}
        <div style={{ 
          background: 'white', 
          borderRadius: '15px', 
          padding: '20px', 
          marginBottom: '20px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ margin: '0 0 10px 0', color: '#333' }}>📌 Agent Connection Status</h3>
              <pre style={{ 
                background: '#f5f5f5', 
                padding: '10px', 
                borderRadius: '5px',
                margin: 0,
                fontSize: '0.9em',
                whiteSpace: 'pre-wrap'
              }}>
                {agentStatus.message}
              </pre>
            </div>
            <button
              onClick={checkAgentStatus}
              style={{
                padding: '10px 20px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '1em'
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

          <div style={{ padding: '30px' }}>
            {/* Upload Tab */}
            {activeTab === 'upload' && (
              <div>
                <h3 style={{ color: '#333', marginBottom: '15px' }}>📤 Upload a Photo</h3>
                <p style={{ color: '#666', marginBottom: '20px' }}>
                  Upload an image showing your upper body and sitting posture. Make sure your shoulders, head, and upper torso are clearly visible from the FRONT.
                </p>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
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
                        padding: '40px',
                        textAlign: 'center',
                        cursor: 'pointer',
                        background: uploadedImage ? '#f0f0f0' : 'white',
                        minHeight: '400px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      {uploadedImage ? (
                        <img src={uploadedImage} alt="Uploaded" style={{ maxWidth: '100%', maxHeight: '350px', borderRadius: '8px' }} />
                      ) : (
                        <>
                          <Upload size={48} color="#667eea" />
                          <p style={{ marginTop: '20px', color: '#666' }}>Click to upload image</p>
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
                        fontSize: '1.1em',
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
                <h3 style={{ color: '#333', marginBottom: '15px' }}>📷 Capture from Webcam</h3>
                <p style={{ color: '#666', marginBottom: '20px' }}>
                  Position yourself 2-3 feet from camera. Face the camera directly. Ensure upper body is visible.
                </p>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  <div>
                    {!webcamImage && (
                      <div style={{
                        border: '2px solid #667eea',
                        borderRadius: '10px',
                        overflow: 'hidden',
                        background: '#000',
                        minHeight: '400px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        {isCameraActive ? (
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            style={{ width: '100%', height: 'auto' }}
                          />
                        ) : (
                          <div style={{ textAlign: 'center', color: 'white' }}>
                            <Camera size={48} />
                            <p style={{ marginTop: '20px' }}>Camera not active</p>
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
                    
                    <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
                      {!isCameraActive && !webcamImage && (
                        <button
                          onClick={startWebcam}
                          style={{
                            flex: 1,
                            padding: '15px',
                            background: '#28a745',
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            fontSize: '1.1em',
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
                              padding: '15px',
                              background: '#667eea',
                              color: 'white',
                              border: 'none',
                              borderRadius: '8px',
                              cursor: 'pointer',
                              fontSize: '1.1em',
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
                              fontSize: '1.1em',
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
                            padding: '15px',
                            background: loading ? '#ccc' : '#667eea',
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: loading ? 'not-allowed' : 'pointer',
                            fontSize: '1.1em',
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
              <div style={{ color: '#333' }}>
                <h2>🎯 How to Get Accurate Results</h2>
                
                <h3 style={{ marginTop: '30px', color: '#667eea' }}>✅ CORRECT Setup (Frontal View)</h3>
                <div style={{ background: '#f5f5f5', padding: '20px', borderRadius: '10px', marginTop: '10px' }}>
                  <pre style={{ margin: 0 }}>{`
        📷 Camera
         ↓
       👤 YOU
========================
|      CHAIR           |
========================
                  `}</pre>
                </div>
                
                <h3 style={{ marginTop: '30px', color: '#667eea' }}>Requirements:</h3>
                <ul style={{ lineHeight: '2' }}>
                  <li>✅ Face camera directly (frontal view)</li>
                  <li>✅ 2-3 feet distance from camera</li>
                  <li>✅ Upper body visible (shoulders + head + torso)</li>
                  <li>✅ Good lighting (in front of you)</li>
                  <li>✅ Camera at eye level</li>
                </ul>
                
                <h3 style={{ marginTop: '30px', color: '#667eea' }}>📊 Understanding Scores</h3>
                <ul style={{ lineHeight: '2' }}>
                  <li><strong>85-100 🟢 Excellent</strong> - Keep it up!</li>
                  <li><strong>70-84 🟡 Good</strong> - Minor adjustments needed</li>
                  <li><strong>50-69 🟠 Fair</strong> - Significant improvement needed</li>
                  <li><strong>0-49 🔴 Poor</strong> - Immediate correction required</li>
                </ul>

                <h3 style={{ marginTop: '30px', color: '#667eea' }}>🤖 AI Analysis Modes</h3>
                <p><strong>Hybrid Mode</strong> (Best):</p>
                <ul>
                  <li>Deep Learning Classification (70% weight)</li>
                  <li>MediaPipe Geometric Analysis (30% weight)</li>
                  <li>Confidence scores and probabilities</li>
                </ul>
                
                <p style={{ marginTop: '15px' }}><strong>MediaPipe-Only Mode</strong> (Fallback):</p>
                <ul>
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
          padding: '20px',
          marginTop: '20px',
          textAlign: 'center',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <p style={{ color: '#666', margin: '10px 0' }}>
            💡 <strong>Tip:</strong> Maintain good posture: Keep your back straight, shoulders relaxed, 
            and head aligned with your spine. Take breaks every 30 minutes!
          </p>
          <p style={{ color: '#999', fontSize: '0.9em', marginTop: '10px' }}>
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
        padding: '30px',
        textAlign: 'center',
        color: 'white',
        marginBottom: '20px'
      }}>
        <div style={{ fontSize: '3em', margin: '0' }}>{scoreInfo.emoji}</div>
        <h2 style={{ margin: '10px 0', fontSize: '2em' }}>Score: {result.score}/100</h2>
        <div style={{
          background: 'white',
          color: scoreInfo.bg,
          display: 'inline-block',
          padding: '10px 30px',
          borderRadius: '25px',
          margin: '10px 0',
          fontWeight: 'bold',
          fontSize: '1.2em'
        }}>
          {scoreInfo.label}
        </div>
        <p style={{ margin: '15px 0', fontSize: '1.1em' }}>Status: <strong>{result.status.toUpperCase()}</strong></p>
      </div>

      <div style={{ 
        background: '#f8f9fa', 
        borderRadius: '10px', 
        padding: '20px',
        maxHeight: '500px',
        overflowY: 'auto'
      }}>
        <h3 style={{ color: '#333', marginTop: 0 }}>💬 Feedback</h3>
        <p style={{ color: '#666', lineHeight: '1.6' }}>{result.feedback}</p>

        <h3 style={{ color: '#333', marginTop: '20px' }}>📊 Metrics</h3>
        <ul style={{ color: '#666', lineHeight: '1.8' }}>
          <li>Spine Angle: {result.metrics.spine_angle}°</li>
          <li>Shoulder Slope: {result.metrics.shoulder_slope}</li>
          <li>Head Forward Distance: {result.metrics.head_forward_distance}</li>
        </ul>

        {result.issues && result.issues.length > 0 && (
          <>
            <h3 style={{ color: '#dc3545', marginTop: '20px' }}>⚠️ Issues Detected</h3>
            <ul style={{ color: '#666', lineHeight: '1.8' }}>
              {result.issues.map((issue, i) => (
                <li key={i}>{issue}</li>
              ))}
            </ul>
          </>
        )}

        {result.dlClassification && (
          <>
            <h3 style={{ color: '#333', marginTop: '20px' }}>🤖 AI Classification</h3>
            <ul style={{ color: '#666', lineHeight: '1.8' }}>
              <li>Predicted Class: <strong>{result.dlClassification.predicted_class.toUpperCase()}</strong></li>
              <li>Confidence: <strong>{(result.dlClassification.confidence * 100).toFixed(1)}%</strong></li>
            </ul>
          </>
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