'use strict';
'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Eye, Upload, Settings, X, ShieldAlert, CheckCircle, Activity, Info, BarChart2, Sun, Moon, Sliders, Trash2, Edit3 } from 'lucide-react';
import styles from './page.module.css';

export default function Home() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [explanation, setExplanation] = useState('');
  const [error, setError] = useState(null);
  
  // Settings - default to local running python server
  const [apiEndpoint, setApiEndpoint] = useState('http://127.0.0.1:7860');
  const [tempEndpoint, setTempEndpoint] = useState('http://127.0.0.1:7860');
  const [showSettings, setShowSettings] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const [theme, setTheme] = useState('dark');

  // Premium Features States
  const [preprocessedPreview, setPreprocessedPreview] = useState(null);
  const [gradcamPreview, setGradcamPreview] = useState(null);
  const [activeTab, setActiveTab] = useState('original'); // 'original', 'preprocessed', 'gradcam'
  const [opacity, setOpacity] = useState(0.5);
  
  // Canvas Annotation States
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushColor, setBrushColor] = useState('#ef4444');
  const [brushSize, setBrushSize] = useState(4);
  const [notes, setNotes] = useState('');
  const [enableAnnotations, setEnableAnnotations] = useState(true);

  // Load theme and endpoint from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);

    const saved = localStorage.getItem('dr_api_endpoint');
    if (saved) {
      setApiEndpoint(saved);
      setTempEndpoint(saved);
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

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
      setPreprocessedPreview(null);
      setGradcamPreview(null);
      setActiveTab('original');
      setNotes('');
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

  const clearAnnotations = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const removeImage = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setExplanation('');
    setError(null);
    setPreprocessedPreview(null);
    setGradcamPreview(null);
    setActiveTab('original');
    setNotes('');
    clearAnnotations();
  };

  // Drawing Handlers
  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    
    if (clientX === undefined || clientY === undefined) return;
    
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    
    if (clientX === undefined || clientY === undefined) return;
    
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.strokeStyle = brushColor;
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const handleImageLoad = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = e.target.clientWidth;
    canvas.height = e.target.clientHeight;
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
        
        if (resJson.data[2]) {
          setPreprocessedPreview(resJson.data[2]);
        }
        if (resJson.data[3]) {
          setGradcamPreview(resJson.data[3]);
          setActiveTab('gradcam');
        }
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
            <li style={{ color: 'var(--foreground)', opacity: 0.8 }}>
              {parts.map((part, pIdx) => pIdx % 2 === 1 ? <strong key={pIdx} style={{ color: 'var(--foreground)' }}>{part}</strong> : part)}
            </li>
          </ul>
        );
      }
      return <p key={idx} style={{ margin: '0.5rem 0', color: 'var(--foreground)', opacity: 0.8 }}>{trimmed}</p>;
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
          </div>

          <div className={styles.headerActions}>
            <button 
              className={styles.themeToggleBtn}
              onClick={toggleTheme}
              title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              aria-label="Toggle Theme"
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button 
              className={styles.themeToggleBtn}
              onClick={() => setShowSettings(true)}
              title="Configure API Endpoint"
              aria-label="Configure API Endpoint"
            >
              <Settings size={18} />
            </button>
          </div>
        </div>
      </header>

      <main className={styles.dashboardGrid}>
        {/* Left Console: Image Acquisition & Work Area */}
        <section className={`${styles.glassPanel} ${styles.colSpan7}`}>
          <div className={styles.panelHeader}>
            <div className={styles.panelTitleGroup}>
              <Activity size={18} className={styles.textPrimary} />
              <h2 className={styles.panelTitle}>Image Acquisition & Annotation</h2>
            </div>
            <p className={styles.panelSubtitle}>
              Acquire fundus photographs and draw/save medical annotations.
            </p>
          </div>

          <div className={styles.workspaceBody}>
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
                <div className={styles.dropzoneVisual}>
                  <Upload className={styles.uploadIcon} size={36} />
                </div>
                <p className={styles.uploadText}>Import Fundus Photograph</p>
                <p className={styles.uploadHint}>Drag and drop retina scan here or click to browse</p>
              </div>
            ) : (
              <div className={styles.previewContainer}>
                <div className={styles.imageTabs}>
                  <button 
                    className={`${styles.tabBtn} ${activeTab === 'original' ? styles.activeTabBtn : ''}`}
                    onClick={() => setActiveTab('original')}
                  >
                    <Eye size={14} />
                    Original / Drawing
                  </button>
                  <button 
                    className={`${styles.tabBtn} ${activeTab === 'preprocessed' ? styles.activeTabBtn : ''}`}
                    onClick={() => setActiveTab('preprocessed')}
                    disabled={!results}
                    title={!results ? 'Run assessment first' : ''}
                  >
                    <Sliders size={14} />
                    Preprocessed View
                  </button>
                  <button 
                    className={`${styles.tabBtn} ${activeTab === 'gradcam' ? styles.activeTabBtn : ''}`}
                    onClick={() => setActiveTab('gradcam')}
                    disabled={!results}
                    title={!results ? 'Run assessment first' : ''}
                  >
                    <Activity size={14} />
                    AI Grad-CAM
                  </button>
                </div>

                <div className={styles.imageDisplayWrapper}>
                  {/* 1. Original Image View with Annotation Overlay */}
                  <div style={{ display: activeTab === 'original' ? 'block' : 'none', position: 'relative', width: '100%', height: '100%' }}>
                    <img 
                      src={preview} 
                      alt="Retinal fundus preview" 
                      className={styles.previewImage} 
                      onLoad={handleImageLoad}
                    />
                    {enableAnnotations && (
                      <canvas
                        ref={canvasRef}
                        className={styles.annotationCanvas}
                        onMouseDown={startDrawing}
                        onMouseMove={draw}
                        onMouseUp={stopDrawing}
                        onMouseLeave={stopDrawing}
                        onTouchStart={startDrawing}
                        onTouchMove={draw}
                        onTouchEnd={stopDrawing}
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          cursor: 'crosshair',
                          pointerEvents: loading ? 'none' : 'auto'
                        }}
                      />
                    )}
                  </div>

                  {/* 2. Preprocessed Image View (Side-by-Side Comparison) */}
                  {activeTab === 'preprocessed' && (
                    preprocessedPreview ? (
                      <div className={styles.comparisonContainer}>
                        <div className={styles.comparisonHalf}>
                          <span className={styles.comparisonLabel}>Original</span>
                          <img src={preview} alt="Original scan" className={styles.previewImage} />
                        </div>
                        <div className={styles.comparisonHalf}>
                          <span className={styles.comparisonLabel}>Graham Filtered</span>
                          <img src={preprocessedPreview} alt="Preprocessed scan" className={styles.previewImage} />
                        </div>
                      </div>
                    ) : (
                      <div className={styles.featureWarningContainer}>
                        <Sliders size={32} className={styles.warningIcon} />
                        <h4>Graham Filter Not Available</h4>
                        <p>Your diagnostic API host only returned 2 outputs. To view Graham local-contrast filtering, please run the updated local Python backend (<code>python app.py</code>) or update your Hugging Face Space files.</p>
                      </div>
                    )
                  )}

                  {/* 3. Grad-CAM Heatmap View with Transparency Blend Control */}
                  {activeTab === 'gradcam' && (
                    gradcamPreview ? (
                      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                        <img src={preview} alt="Original scan" className={styles.previewImage} />
                        <img 
                          src={gradcamPreview} 
                          alt="Grad-CAM heatmap" 
                          className={styles.previewImage} 
                          style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            opacity: opacity,
                            mixBlendMode: 'normal',
                            transition: 'opacity 0.15s ease',
                            pointerEvents: 'none'
                          }}
                        />
                        
                        <div className={styles.opacitySliderContainer}>
                          <span className={styles.sliderLabel}>Overlay Opacity: {Math.round(opacity * 100)}%</span>
                          <input 
                            type="range" 
                            min="0.1" 
                            max="0.9" 
                            step="0.05" 
                            value={opacity} 
                            onChange={(e) => setOpacity(parseFloat(e.target.value))}
                            className={styles.opacitySlider}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className={styles.featureWarningContainer}>
                        <Activity size={32} className={styles.warningIcon} />
                        <h4>AI Feature Maps Not Available</h4>
                        <p>Your diagnostic API host does not support Grad-CAM visualization. To enable feature highlight mapping, run the updated local Python backend (<code>python app.py</code>) or update your Hugging Face Space files.</p>
                      </div>
                    )
                  )}

                  {/* Close Button inside imageDisplayWrapper */}
                  <button className={styles.removeImageBtn} onClick={removeImage} title="Remove image">
                    <X size={16} />
                  </button>
                </div>

                {/* Annotation Control Bar */}
                {activeTab === 'original' && (
                  <div className={styles.annotatorToolbar}>
                    <div className={styles.toolbarGroup}>
                      <button 
                        className={`${styles.toolBtn} ${enableAnnotations ? styles.toolBtnActive : ''}`}
                        onClick={() => setEnableAnnotations(!enableAnnotations)}
                        title={enableAnnotations ? 'Disable annotations' : 'Enable annotations'}
                      >
                        <Edit3 size={14} />
                        {enableAnnotations ? 'Annotate Mode' : 'View Mode'}
                      </button>
                      {enableAnnotations && (
                        <>
                          <div className={styles.colorPalette}>
                            {['#ef4444', '#f59e0b', '#10b981', '#3b82f6'].map(color => (
                              <button
                                key={color}
                                className={`${styles.colorSwatch} ${brushColor === color ? styles.colorSwatchActive : ''}`}
                                style={{ backgroundColor: color }}
                                onClick={() => setBrushColor(color)}
                              />
                            ))}
                          </div>
                          <div className={styles.sizeControl}>
                            <span className={styles.sizeLabel}>{brushSize}px</span>
                            <input 
                              type="range" 
                              min="2" 
                              max="12" 
                              value={brushSize} 
                              onChange={(e) => setBrushSize(parseInt(e.target.value))}
                              className={styles.sizeSlider}
                            />
                          </div>
                        </>
                      )}
                    </div>
                    {enableAnnotations && (
                      <button className={styles.clearBtn} onClick={clearAnnotations}>
                        <Trash2 size={14} />
                        Clear
                      </button>
                    )}
                  </div>
                )}

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

            {/* Clinician Notes Textarea */}
            {preview && (
              <div className={styles.notesContainer}>
                <label className={styles.notesLabel}>Clinician Observations</label>
                <textarea
                  className={styles.notesTextarea}
                  placeholder="Record patient notes, lesion shapes, or vascular defects..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </div>
            )}

            {preview && (
              <button 
                className={styles.analyzeBtn}
                onClick={runAnalysis}
                disabled={loading}
              >
                <Activity size={18} className={loading ? 'animate-spin' : ''} />
                {loading ? 'Analyzing Retinal Structures...' : 'Compute Diagnostic Assessment'}
              </button>
            )}

            {error && (
              <div className={styles.errorAlert}>
                <ShieldAlert size={20} className={styles.errorIcon} />
                <div>
                  <strong className={styles.errorTitle}>Console Sync Failure</strong>
                  <span className={styles.errorText}>{error}</span>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Right Console: Diagnostic Engine Results */}
        <section className={`${styles.glassPanel} ${styles.colSpan5}`}>
          <div className={styles.panelHeader}>
            <div className={styles.panelTitleGroup}>
              <BarChart2 size={18} className={styles.textPrimary} />
              <h2 className={styles.panelTitle}>Diagnostic Intelligence</h2>
            </div>
            <p className={styles.panelSubtitle}>
              Real-time classification probability and clinical explanations.
            </p>
          </div>

          <div className={styles.workspaceBody}>
            {loading ? (
              <div className={styles.loaderContainer}>
                <div className={styles.spinner}>
                  <div className={styles.spinnerRing} />
                  <div className={styles.spinnerRingOuter} />
                </div>
                <h3 className={styles.loaderTitle}>Processing Fundus Scan</h3>
                <p className={styles.loaderText}>Extracting spatial features from neural layers...</p>
              </div>
            ) : results ? (
              <div className={styles.resultsContainer}>
                {/* Unified primary pill result indicator */}
                <div 
                  className={styles.primaryResultCard} 
                  style={{ 
                    borderColor: getSeverityStyle(topPrediction.className).color,
                    boxShadow: theme === 'dark' ? `0 0 20px -5px ${getSeverityStyle(topPrediction.className).color}50` : 'none'
                  }}
                >
                  <div className={styles.indicatorPulse} style={{ backgroundColor: getSeverityStyle(topPrediction.className).color }} />
                  <div className={styles.primaryTextGroup}>
                    <span className={styles.primaryClassLabel}>Primary Classification Match</span>
                    <h3 className={styles.primaryClassName} style={{ color: getSeverityStyle(topPrediction.className).color }}>
                      {topPrediction.className.replace('_', ' ')}
                    </h3>
                  </div>
                  <span className={styles.primaryClassProbability} style={{ color: getSeverityStyle(topPrediction.className).color }}>
                    {(topPrediction.probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className={styles.chartContainer}>
                  {Object.entries(results).map(([className, probability]) => {
                    const style = getSeverityStyle(className);
                    const isTop = topPrediction && topPrediction.className === className;
                    return (
                      <div key={className} className={styles.chartRow} style={{ opacity: isTop ? 1 : 0.65 }}>
                        <div className={styles.chartLabels}>
                          <span className={styles.className} style={{ fontWeight: isTop ? '700' : '400' }}>
                            {className.replace('_', ' ')}
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
                              background: style.gradient,
                              boxShadow: isTop ? `0 0 10px ${style.color}50` : 'none'
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
                <div className={styles.emptyVisual}>
                  <Eye className={styles.emptyIcon} size={42} />
                </div>
                <h3>Diagnostic Monitor Idle</h3>
                <p>
                  Upload and analyze a fundus photograph to engage neural classification mapping.
                </p>
              </div>
            )}
          </div>
        </section>
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
