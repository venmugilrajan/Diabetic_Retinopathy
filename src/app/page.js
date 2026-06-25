'use strict';
'use client';

import React, { useState, useEffect } from 'react';
import { Eye, Upload, Settings, X, ShieldAlert, CheckCircle, Activity, Info, BarChart2 } from 'lucide-react';
import styles from './page.module.css';

export default function Home() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [explanation, setExplanation] = useState('');
  const [error, setError] = useState(null);
  
  // Settings - default to user's Hugging Face Space
  const [apiEndpoint, setApiEndpoint] = useState('venmugilrajan/Diabetic_Retinopathy');
  const [tempEndpoint, setTempEndpoint] = useState('venmugilrajan/Diabetic_Retinopathy');
  const [showSettings, setShowSettings] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);

  // Load endpoint from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('dr_api_endpoint');
    if (saved) {
      setApiEndpoint(saved);
      setTempEndpoint(saved);
    }
  }, []);

  const saveSettings = () => {
    localStorage.setItem('dr_api_endpoint', tempEndpoint);
    setApiEndpoint(tempEndpoint);
    setShowSettings(false);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const processFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
      setImage(e.target.result);
      setResults(null);
      setExplanation('');
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const removeImage = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setExplanation('');
    setError(null);
  };

  const runAnalysis = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image,
          apiEndpoint,
        }),
      });

      const resJson = await response.json();

      if (!response.ok) {
        throw new Error(resJson.error || 'Prediction failed');
      }

      if (resJson && resJson.data && resJson.data.length >= 2) {
        // Extract output structures
        const rawResults = resJson.data[0];
        
        // Format prediction results dict
        let probDict = {};
        if (rawResults && rawResults.confidences) {
          rawResults.confidences.forEach(item => {
            probDict[item.label] = item.confidence;
          });
        } else if (typeof rawResults === 'object') {
          probDict = rawResults;
        }
        
        setResults(probDict);
        setExplanation(resJson.data[1]);
      } else {
        throw new Error('Unexpected response format from backend API.');
      }
    } catch (err) {
      console.error(err);
      let errMsg = err.message || 'Unknown connection error';
      
      if (apiEndpoint.includes('localhost') || apiEndpoint.includes('127.0.0.1')) {
        errMsg = `Failed to connect to local backend at "${apiEndpoint}". Make sure your Python server is running (python app.py) and CORS is enabled. Details: ${errMsg}`;
      } else {
        errMsg = `Failed to connect to Hugging Face Space "${apiEndpoint}". Details: ${errMsg}. Please check if the space is active, public, and running.`;
      }
      setError(errMsg);
    } finally {
      setLoading(false);
    }
  };

  // Helper to parse Gradio server markdown explanations into clean React components
  const renderExplanation = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      let trimmed = line.trim();
      if (trimmed.startsWith('🔬')) {
        return <h3 key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.25rem', margin: '1rem 0 0.5rem 0', color: '#0ea5e9' }}>{trimmed}</h3>;
      }
      if (trimmed.startsWith('**Clinical signs:**')) {
        return <p key={idx} style={{ fontWeight: '600', margin: '0.75rem 0 0.25rem 0' }}>{trimmed.replace(/\*\*/g, '')}</p>;
      }
      if (trimmed.startsWith('*')) {
        // Handle bolding in list items
        const itemText = trimmed.substring(1).trim();
        const parts = itemText.split('**');
        return (
          <ul key={idx} style={{ listStyleType: 'disc', paddingLeft: '1.25rem', margin: '0.25rem 0' }}>
            <li style={{ color: '#a1a1aa' }}>
              {parts.map((part, pIdx) => pIdx % 2 === 1 ? <strong key={pIdx} style={{ color: '#fff' }}>{part}</strong> : part)}
            </li>
          </ul>
        );
      }
      return <p key={idx} style={{ margin: '0.5rem 0', color: '#a1a1aa' }}>{trimmed}</p>;
    });
  };

  // Helper to color code diagnostic severity levels
  const getSeverityStyle = (name) => {
    switch (name) {
      case 'No_DR':
        return { color: '#10b981', gradient: 'linear-gradient(90deg, #059669, #10b981)' };
      case 'Mild':
        return { color: '#eab308', gradient: 'linear-gradient(90deg, #ca8a04, #eab308)' };
      case 'Moderate':
        return { color: '#f97316', gradient: 'linear-gradient(90deg, #ea580c, #f97316)' };
      case 'Severe':
        return { color: '#ef4444', gradient: 'linear-gradient(90deg, #dc2626, #ef4444)' };
      case 'Proliferate_DR':
        return { color: '#ec4899', gradient: 'linear-gradient(90deg, #db2777, #ec4899)' };
      default:
        return { color: '#0ea5e9', gradient: 'linear-gradient(90deg, #0284c7, #0ea5e9)' };
    }
  };

  // Find max percentage to highlight prediction
  const getTopPrediction = () => {
    if (!results) return null;
    let maxVal = -1;
    let topClass = '';
    Object.entries(results).forEach(([key, val]) => {
      if (val > maxVal) {
        maxVal = val;
        topClass = key;
      }
    });
    return { className: topClass, probability: maxVal };
  };

  const topPrediction = getTopPrediction();

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.logoArea}>
            <Eye className={styles.logoIcon} size={28} />
            <span className={styles.logoText}>RetinaScan AI</span>
            <span className={styles.badge}>v1.0</span>
          </div>
          <div className={styles.headerActions}>
            <button 
              className={styles.settingsBtn}
              onClick={() => setShowSettings(true)}
            >
              <Settings size={16} />
              Backend API
            </button>
          </div>
        </div>
      </header>

      <main className={styles.main}>
        {/* Left Side: Upload & Action Panel */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Retinal Image Analyzer</h2>
            <p className={styles.cardSubtitle}>
              Upload fundus photographs to automatically classify Diabetic Retinopathy severity.
            </p>
          </div>

          {!preview ? (
            <div 
              className={`${styles.dropzone} ${isDragActive ? styles.dropzoneActive : ''}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById('fileInput').click()}
            >
              <input 
                id="fileInput"
                type="file"
                className="hidden"
                style={{ display: 'none' }}
                accept="image/*"
                onChange={handleFileChange}
              />
              <Upload className={styles.uploadIcon} size={48} />
              <p className={styles.uploadText}>Drag & drop retinal scan here</p>
              <p className={styles.uploadHint}>or click to browse from device (JPEG, PNG)</p>
            </div>
          ) : (
            <div className={styles.previewContainer}>
              <img src={preview} alt="Retinal fundus preview" className={styles.previewImage} />
              <button className={styles.removeImageBtn} onClick={removeImage} title="Remove image">
                <X size={16} />
              </button>
              {loading && (
                <div className={styles.scanOverlay}>
                  <div className={styles.scanLine} />
                  <div className={styles.retinalTarget}>
                    <div className={styles.retinalTargetInner} />
                  </div>
                </div>
              )}
            </div>
          )}

          {preview && (
            <button 
              className={styles.analyzeBtn}
              onClick={runAnalysis}
              disabled={loading}
            >
              <Activity size={18} className={loading ? 'animate-spin' : ''} />
              {loading ? 'Analyzing Retinal Structures...' : 'Start Diagnostic Assessment'}
            </button>
          )}

          {error && (
            <div style={{ marginTop: '1rem', padding: '1rem', borderRadius: 'var(--radius)', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: '#ef4444', display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
              <ShieldAlert size={20} style={{ flexShrink: 0, marginTop: '0.1rem' }} />
              <div>
                <strong style={{ display: 'block', marginBottom: '0.25rem' }}>Connection Failed</strong>
                <span style={{ fontSize: '0.85rem' }}>{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Diagnostics & Breakdown */}
        <div className={styles.card}>
          {loading ? (
            <div className={styles.loaderContainer}>
              <div className={styles.spinner}>
                <div className={styles.spinnerRing} />
                <div className={styles.spinnerRingOuter} />
              </div>
              <h3 className={styles.loaderTitle}>Processing Fundus Scan</h3>
              <p className={styles.loaderText}>Executing Ben Graham preprocessing & feeding model...</p>
            </div>
          ) : results ? (
            <div className={styles.resultsContainer}>
              <div className={styles.resultHeader}>
                <h3 className={styles.resultTitle}>Diagnostic Results</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: '600', color: getSeverityStyle(topPrediction.className).color }}>
                  <CheckCircle size={18} />
                  <span>Detected Stage: {topPrediction.className.replace('_', ' ')}</span>
                </div>
              </div>

              <div className={styles.chartContainer}>
                {Object.entries(results).map(([className, probability]) => {
                  const style = getSeverityStyle(className);
                  const isTop = topPrediction && topPrediction.className === className;
                  return (
                    <div key={className} className={styles.chartRow} style={{ opacity: isTop ? 1 : 0.6 }}>
                      <div className={styles.chartLabels}>
                        <span className={styles.className} style={{ fontWeight: isTop ? '700' : '400' }}>
                          {className.replace('_', ' ')} {isTop && ' (Primary Match)'}
                        </span>
                        <span className={styles.classPercentage} style={{ color: style.color }}>
                          {(probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className={styles.progressBarContainer}>
                        <div 
                          className={styles.progressBar} 
                          style={{ 
                            width: `${probability * 100}%`,
                            background: style.gradient
                          }} 
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {explanation && (
                <div className={styles.explanationBox}>
                  {renderExplanation(explanation)}
                </div>
              )}
            </div>
          ) : (
            <div className={styles.emptyState}>
              <BarChart2 className={styles.emptyIcon} size={64} />
              <h3>Awaiting Input</h3>
              <p style={{ fontSize: '0.875rem', marginTop: '0.5rem', maxWidth: '320px' }}>
                Upload and submit a retinal fundus scan on the left to see the severity score analysis.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Settings Modal */}
      {showSettings && (
        <div className={styles.modalOverlay}>
          <div className={styles.modal}>
            <h3 className={styles.modalTitle}>Diagnostic Backend Connection</h3>
            <div className={styles.inputGroup}>
              <label className={styles.inputLabel}>Gradio Server API URL</label>
              <input 
                type="text" 
                className={styles.textInput}
                value={tempEndpoint}
                onChange={(e) => setTempEndpoint(e.target.value)}
                placeholder="e.g. http://localhost:7860"
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--muted-foreground)' }}>
                Set this to your running python app.py address or a Hugging Face Space URL.
              </span>
            </div>
            <div className={styles.modalActions}>
              <button className={styles.btnSecondary} onClick={() => setShowSettings(false)}>
                Cancel
              </button>
              <button className={styles.btnPrimary} onClick={saveSettings}>
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
