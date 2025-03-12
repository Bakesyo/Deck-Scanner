// Project structure overview:
//
// ├── public/
// │   ├── manifest.json         # PWA manifest
// │   ├── service-worker.js     # Service worker for offline functionality
// │   ├── models/               # ML model files
// │   │   ├── model.json        # TensorFlow.js model config
// │   │   ├── weights.bin       # Model weights
// │   │   └── labels.json       # Deck classification labels
// │   └── icons/                # App icons for various devices
// ├── src/
// │   ├── components/           # React components
// │   ├── services/             # Core services (scanner, database)
// │   ├── hooks/                # Custom React hooks
// │   ├── utils/                # Utility functions
// │   ├── App.js                # Main application component
// │   └── index.js              # Entry point
// ├── netlify.toml              # Netlify configuration
// └── package.json              # Dependencies and scripts

// ------------------------------------------------------
// src/services/DeckScannerService.js
// ------------------------------------------------------

import * as tf from '@tensorflow/tfjs';
import { createWorker } from 'tesseract.js';
import { v4 as uuidv4 } from 'uuid';
import { DatabaseService } from './DatabaseService';

/**
 * DeckScannerService - Provides browser-compatible card deck recognition
 * Optimized for mobile PWA deployment with offline-first functionality
 */
class DeckScannerService {
  constructor() {
    this.model = null;
    this.isModelLoaded = false;
    this.processingLock = false;
    this.scanResults = [];
    this.dbService = new DatabaseService();
    this.ocrWorker = null;
    this.modelUrl = './models/model.json';
    this.labelUrl = './models/labels.json';
    this.labels = [];
  }

  /**
   * Initialize the scanner service
   * @returns {Promise<boolean>} Initialization status
   */
  async initialize() {
    try {
      await this.dbService.initialize();
      
      // Load TensorFlow.js model
      this.model = await tf.loadGraphModel(this.modelUrl);
      
      // Load classification labels
      const labelsResponse = await fetch(this.labelUrl);
      this.labels = await labelsResponse.json();
      
      // Initialize OCR worker for text recognition on cards
      this.ocrWorker = createWorker({
        langPath: './tessdata',
        logger: m => console.debug(m),
        errorHandler: err => console.error(err)
      });
      
      await this.ocrWorker.load();
      await this.ocrWorker.loadLanguage('eng');
      await this.ocrWorker.initialize('eng');
      
      this.isModelLoaded = true;
      return true;
    } catch (error) {
      console.error('Scanner initialization failed:', error);
      return false;
    }
  }
  
  /**
   * Process image for deck recognition
   * @param {HTMLImageElement|ImageData} imageData Image to process
   * @returns {Promise<RecognitionResult>} Recognition result
   */
  async processImage(imageData) {
    if (!this.isModelLoaded) {
      throw new Error('Scanner not initialized');
    }
    
    try {
      // Convert input to tensor
      let tensor;
      if (imageData instanceof HTMLImageElement) {
        tensor = tf.browser.fromPixels(imageData);
      } else {
        tensor = tf.tensor(imageData.data, [imageData.height, imageData.width, 4]);
      }
      
      // Preprocess image
      const preprocessed = this.preprocessImage(tensor);
      
      // Run inference
      const predictions = await this.model.predict(preprocessed);
      const resultsArray = await predictions.data();
      
      // Get top prediction
      const topIdx = this.getTopPredictionIndex(resultsArray);
      const deckInfo = this.labels[topIdx];
      const confidence = resultsArray[topIdx];
      
      // Extract text from image for additional verification
      const { text } = await this.ocrWorker.recognize(imageData);
      const textVerification = this.verifyTextResults(text, deckInfo);
      
      // Get pricing information
      const pricing = await this.dbService.getPricingData(deckInfo.id);
      
      // Cleanup tensors
      tensor.dispose();
      preprocessed.dispose();
      predictions.dispose();
      
      // Return complete result
      return {
        deckId: deckInfo.id,
        deckName: deckInfo.name,
        manufacturer: deckInfo.manufacturer,
        casino: deckInfo.casino,
        confidence: confidence,
        textVerification: textVerification,
        pricing: pricing,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Image processing error:', error);
      throw error;
    }
  }
  
  /**
   * Start batch scanning session
   * @param {Function} onDeckIdentified Callback for results
   * @returns {string} Session ID
   */
  startBatchScanning(onDeckIdentified) {
    const sessionId = uuidv4();
    this.scanResults = [];
    this.batchActive = true;
    
    // Return session ID for tracking
    return sessionId;
  }
  
  /**
   * Process a frame from video stream
   * @param {HTMLVideoElement} videoElement Video element
   * @param {string} sessionId Active session ID
   * @param {Function} onResult Callback for results
   */
  async processVideoFrame(videoElement, sessionId, onResult) {
    if (!this.batchActive) return;
    if (this.processingLock) return;
    
    this.processingLock = true;
    
    try {
      // Create canvas from video frame
      const canvas = document.createElement('canvas');
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoElement, 0, 0);
      
      // Process image
      const result = await this.processImage(canvas);
      
      // Add to results if confidence is high enough
      if (result.confidence > 0.75) {
        const scanRecord = {
          id: uuidv4(),
          sessionId: sessionId,
          deckId: result.deckId,
          timestamp: new Date().toISOString(),
          confidence: result.confidence,
          pricingSnapshot: {
            buyPrice: result.pricing.buyPrice,
            sellPrice: result.pricing.sellPrice
          }
        };
        
        // Save scan record to database
        await this.dbService.saveScanRecord(scanRecord);
        
        // Add to current batch results
        this.scanResults.push(result);
        
        // Notify via callback
        onResult(result);
      }
    } catch (error) {
      console.error('Video frame processing error:', error);
    } finally {
      this.processingLock = false;
    }
  }
  
  /**
   * Stop active batch scanning session
   * @returns {BatchScanSummary} Batch scanning summary
   */
  stopBatchScanning() {
    this.batchActive = false;
    
    // Calculate summary statistics
    const summary = {
      totalDecks: this.scanResults.length,
      totalBuyValue: this.scanResults.reduce((sum, r) => sum + r.pricing.buyPrice, 0).toFixed(2),
      totalSellValue: this.scanResults.reduce((sum, r) => sum + r.pricing.sellPrice, 0).toFixed(2),
      totalProfit: this.scanResults.reduce((sum, r) => 
        sum + (r.pricing.sellPrice - r.pricing.buyPrice), 0).toFixed(2),
      averageMargin: this.scanResults.length > 0 ? 
        (this.scanResults.reduce((sum, r) => 
          sum + ((r.pricing.sellPrice - r.pricing.buyPrice) / r.pricing.buyPrice * 100), 0) 
          / this.scanResults.length).toFixed(1) + '%' : '0%',
      mostProfitable: this.scanResults.length > 0 ?
        this.scanResults.reduce((max, r) => 
          (r.pricing.sellPrice - r.pricing.buyPrice) > 
          (max.pricing.sellPrice - max.pricing.buyPrice) ? r : max, 
          this.scanResults[0]) : null
    };
    
    return {
      results: this.scanResults,
      summary: summary
    };
  }
  
  /**
   * Export scan results
   * @param {string} format Export format (csv, json)
   * @returns {Promise<Blob>} Export data blob
   */
  async exportResults(format = 'json') {
    if (format === 'csv') {
      return this.exportAsCSV();
    } else {
      return this.exportAsJSON();
    }
  }
  
  // ---------- Private methods ----------
  
  /**
   * Preprocess image for the model
   * @param {tf.Tensor3D} tensor Image tensor
   * @returns {tf.Tensor4D} Preprocessed tensor
   * @private
   */
  preprocessImage(tensor) {
    // Resize to model input size
    const resized = tf.image.resizeBilinear(tensor, [224, 224]);
    
    // Normalize pixel values to [-1, 1]
    const normalized = resized.div(tf.scalar(127.5)).sub(tf.scalar(1));
    
    // Expand dimensions to create batch of 1
    const batched = normalized.expandDims(0);
    
    return batched;
  }
  
  /**
   * Get index of top prediction
   * @param {Float32Array} predictions Prediction array
   * @returns {number} Index of top prediction
   * @private
   */
  getTopPredictionIndex(predictions) {
    let maxIdx = 0;
    let maxVal = predictions[0];
    
    for (let i = 1; i < predictions.length; i++) {
      if (predictions[i] > maxVal) {
        maxVal = predictions[i];
        maxIdx = i;
      }
    }
    
    return maxIdx;
  }
  
  /**
   * Verify OCR text results against expected deck info
   * @param {string} text Recognized text
   * @param {Object} deckInfo Expected deck info
   * @returns {Object} Verification results
   * @private
   */
  verifyTextResults(text, deckInfo) {
    const textLower = text.toLowerCase();
    const manufacturerFound = textLower.includes(deckInfo.manufacturer.toLowerCase());
    const casinoFound = deckInfo.casino ? 
      textLower.includes(deckInfo.casino.toLowerCase()) : false;
    
    return {
      manufacturerVerified: manufacturerFound,
      casinoVerified: casinoFound,
      verificationScore: manufacturerFound ? (casinoFound ? 1.0 : 0.7) : 0.3
    };
  }
  
  /**
   * Export results as CSV
   * @returns {Promise<Blob>} CSV blob
   * @private
   */
  async exportAsCSV() {
    const header = 'Deck Name,Manufacturer,Casino,Buy Price,Sell Price,Profit,Margin %,Confidence,Timestamp\n';
    
    const rows = this.scanResults.map(r => {
      const profit = (r.pricing.sellPrice - r.pricing.buyPrice).toFixed(2);
      const marginPct = ((r.pricing.sellPrice - r.pricing.buyPrice) / 
                        r.pricing.buyPrice * 100).toFixed(1);
      
      return `"${r.deckName}","${r.manufacturer}","${r.casino || ''}",` +
             `${r.pricing.buyPrice.toFixed(2)},${r.pricing.sellPrice.toFixed(2)},` +
             `${profit},${marginPct}%,${(r.confidence * 100).toFixed(1)}%,` +
             `${r.timestamp}`;
    }).join('\n');
    
    const csvContent = header + rows;
    return new Blob([csvContent], { type: 'text/csv' });
  }
  
  /**
   * Export results as JSON
   * @returns {Promise<Blob>} JSON blob
   * @private
   */
  async exportAsJSON() {
    const jsonContent = JSON.stringify({
      exportDate: new Date().toISOString(),
      results: this.scanResults
    }, null, 2);
    
    return new Blob([jsonContent], { type: 'application/json' });
  }
}

export default DeckScannerService;

// ------------------------------------------------------
// src/services/DatabaseService.js
// ------------------------------------------------------

/**
 * DatabaseService - Provides IndexedDB storage for the deck scanner
 * Includes offline-first data persistence and synchronization
 */
class DatabaseService {
  constructor() {
    this.db = null;
    this.DB_NAME = 'deck_scanner_db';
    this.DB_VERSION = 1;
    this.STORES = {
      DECKS: 'decks',
      PRICING: 'pricing',
      SCAN_HISTORY: 'scan_history',
      SYNC_QUEUE: 'sync_queue'
    };
  }
  
  /**
   * Initialize database
   * @returns {Promise<boolean>} Initialization status
   */
  async initialize() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);
      
      request.onerror = event => {
        console.error('Database error:', event.target.error);
        reject(event.target.error);
      };
      
      request.onsuccess = event => {
        this.db = event.target.result;
        resolve(true);
      };
      
      request.onupgradeneeded = event => {
        const db = event.target.result;
        
        // Create object stores
        if (!db.objectStoreNames.contains(this.STORES.DECKS)) {
          const deckStore = db.createObjectStore(this.STORES.DECKS, { keyPath: 'deckId' });
          deckStore.createIndex('manufacturer', 'manufacturer', { unique: false });
          deckStore.createIndex('casino', 'casino', { unique: false });
          deckStore.createIndex('visualHash', 'visualData.backImageHash', { unique: true });
        }
        
        if (!db.objectStoreNames.contains(this.STORES.PRICING)) {
          const pricingStore = db.createObjectStore(this.STORES.PRICING, { keyPath: 'id' });
          pricingStore.createIndex('deckId', 'deckId', { unique: true });
          pricingStore.createIndex('lastUpdated', 'metadata.lastUpdated', { unique: false });
        }
        
        if (!db.objectStoreNames.contains(this.STORES.SCAN_HISTORY)) {
          const historyStore = db.createObjectStore(this.STORES.SCAN_HISTORY, { keyPath: 'id' });
          historyStore.createIndex('sessionId', 'sessionId', { unique: false });
          historyStore.createIndex('timestamp', 'timestamp', { unique: false });
          historyStore.createIndex('deckId', 'deckId', { unique: false });
        }
        
        if (!db.objectStoreNames.contains(this.STORES.SYNC_QUEUE)) {
          db.createObjectStore(this.STORES.SYNC_QUEUE, { 
            keyPath: 'id', 
            autoIncrement: true 
          });
        }
        
        // Load initial data
        this.loadInitialData(db);
      };
    });
  }
  
  /**
   * Load initial deck and pricing data
   * @param {IDBDatabase} db Database instance
   * @private
   */
  async loadInitialData(db) {
    try {
      // Fetch initial data from assets
      const response = await fetch('./data/initial_data.json');
      const data = await response.json();
      
      // Populate decks
      const deckTx = db.transaction(this.STORES.DECKS, 'readwrite');
      const deckStore = deckTx.objectStore(this.STORES.DECKS);
      
      for (const deck of data.decks) {
        deckStore.add(deck);
      }
      
      // Populate pricing
      const pricingTx = db.transaction(this.STORES.PRICING, 'readwrite');
      const pricingStore = pricingTx.objectStore(this.STORES.PRICING);
      
      for (const pricing of data.pricing) {
        pricingStore.add(pricing);
      }
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  }
  
  /**
   * Get pricing data for a deck
   * @param {string} deckId Deck ID
   * @returns {Promise<Object>} Pricing data
   */
  async getPricingData(deckId) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORES.PRICING, 'readonly');
      const store = tx.objectStore(this.STORES.PRICING);
      const index = store.index('deckId');
      
      const request = index.get(deckId);
      
      request.onsuccess = event => {
        const result = event.target.result;
        
        if (result) {
          // Check if data is stale (older than 24 hours)
          const lastUpdated = new Date(result.metadata.lastUpdated);
          const now = new Date();
          const hoursSinceUpdate = (now - lastUpdated) / (1000 * 60 * 60);
          
          if (hoursSinceUpdate > 24) {
            // Queue for background sync
            this.queuePricingSyncTask(deckId);
          }
          
          resolve(result);
        } else {
          // Default pricing if not found
          resolve({
            id: `default_${deckId}`,
            deckId: deckId,
            buyPrice: 0,
            sellPrice: 0,
            metadata: {
              lastUpdated: new Date().toISOString(),
              confidenceScore: 0,
              dataSource: 'default'
            }
          });
        }
      };
      
      request.onerror = event => {
        reject(event.target.error);
      };
    });
  }
  
  /**
   * Save scan record
   * @param {Object} scanRecord Scan record to save
   * @returns {Promise<boolean>} Success status
   */
  async saveScanRecord(scanRecord) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORES.SCAN_HISTORY, 'readwrite');
      const store = tx.objectStore(this.STORES.SCAN_HISTORY);
      
      const request = store.add(scanRecord);
      
      request.onsuccess = () => {
        // Queue for sync with cloud
        this.queueScanHistorySyncTask(scanRecord.id);
        resolve(true);
      };
      
      request.onerror = event => {
        reject(event.target.error);
      };
    });
  }
  
  /**
   * Get scan history for a session
   * @param {string} sessionId Session ID
   * @returns {Promise<Array>} Scan records
   */
  async getSessionHistory(sessionId) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORES.SCAN_HISTORY, 'readonly');
      const store = tx.objectStore(this.STORES.SCAN_HISTORY);
      const index = store.index('sessionId');
      
      const request = index.getAll(sessionId);
      
      request.onsuccess = event => {
        resolve(event.target.result);
      };
      
      request.onerror = event => {
        reject(event.target.error);
      };
    });
  }
  
  /**
   * Queue pricing data for background sync
   * @param {string} deckId Deck ID
   * @private
   */
  async queuePricingSyncTask(deckId) {
    const tx = this.db.transaction(this.STORES.SYNC_QUEUE, 'readwrite');
    const store = tx.objectStore(this.STORES.SYNC_QUEUE);
    
    store.add({
      type: 'pricing_sync',
      deckId: deckId,
      timestamp: new Date().toISOString(),
      status: 'pending'
    });
  }
  
  /**
   * Queue scan history for background sync
   * @param {string} recordId Record ID
   * @private
   */
  async queueScanHistorySyncTask(recordId) {
    const tx = this.db.transaction(this.STORES.SYNC_QUEUE, 'readwrite');
    const store = tx.objectStore(this.STORES.SYNC_QUEUE);
    
    store.add({
      type: 'history_sync',
      recordId: recordId,
      timestamp: new Date().toISOString(),
      status: 'pending'
    });
  }
  
  /**
   * Process background sync queue
   * Should be called periodically when online
   * @returns {Promise<number>} Number of processed items
   */
  async processSyncQueue() {
    // Implementation for background sync
    // Would connect to cloud service when online
  }
}

export { DatabaseService };

// ------------------------------------------------------
// src/components/ScannerView.jsx
// ------------------------------------------------------

import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import DeckScannerService from '../services/DeckScannerService';

/**
 * Scanner component - Provides UI for deck scanning
 */
const ScannerView = () => {
  const [initializing, setInitializing] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [results, setResults] = useState([]);
  const [cameraPermission, setCameraPermission] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');
  
  const videoRef = useRef(null);
  const scannerRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  
  const navigate = useNavigate();
  
  // Initialize scanner
  useEffect(() => {
    const initScanner = async () => {
      scannerRef.current = new DeckScannerService();
      try {
        await scannerRef.current.initialize();
        setInitializing(false);
      } catch (error) {
        console.error('Failed to initialize scanner:', error);
        setCameraError('Failed to initialize scanner');
      }
    };
    
    initScanner();
    
    return () => {
      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Request camera access
  const requestCameraAccess = async () => {
    try {
      const constraints = {
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      setCameraPermission(true);
      setCameraError(null);
    } catch (error) {
      console.error('Camera access error:', error);
      setCameraError(`Camera access denied: ${error.message}`);
    }
  };
  
  // Toggle camera facing mode
  const toggleCamera = async () => {
    // Stop current stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    
    // Switch facing mode
    setFacingMode(facingMode === 'environment' ? 'user' : 'environment');
    
    // Request camera with new facing mode
    await requestCameraAccess();
  };
  
  // Start scanning
  const startScanning = () => {
    if (!scannerRef.current || !videoRef.current) return;
    
    const newSessionId = scannerRef.current.startBatchScanning(result => {
      setResults(prev => [result, ...prev]);
    });
    
    setSessionId(newSessionId);
    setScanning(true);
    setResults([]);
    
    // Start processing frames
    const processFrame = () => {
      if (videoRef.current && scannerRef.current && scanning) {
        scannerRef.current.processVideoFrame(
          videoRef.current, 
          newSessionId,
          result => {
            // This callback is handled by the batch scanning callback
          }
        );
        
        animationRef.current = requestAnimationFrame(processFrame);
      }
    };
    
    animationRef.current = requestAnimationFrame(processFrame);
  };
  
  // Stop scanning
  const stopScanning = () => {
    setScanning(false);
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    if (scannerRef.current) {
      const summary = scannerRef.current.stopBatchScanning();
      navigate('/summary', { state: { summary } });
    }
  };
  
  return (
    <div className="scanner-view">
      <header className="scanner-header">
        <h1>Deck Scanner</h1>
        {scanning && (
          <div className="scan-stats">
            <span className="decks-found">{results.length} decks</span>
            <span className="total-value">
              ${results.reduce((sum, r) => sum + r.pricing.sellPrice, 0).toFixed(2)}
            </span>
          </div>
        )}
      </header>
      
      <main className="scanner-main">
        {initializing ? (
          <div className="initializing">
            <div className="spinner"></div>
            <p>Initializing scanner...</p>
          </div>
        ) : (
          <>
            {!cameraPermission ? (
              <div className="camera-permission">
                <button className="camera-button" onClick={requestCameraAccess}>
                  Enable Camera
                </button>
                {cameraError && <p className="error">{cameraError}</p>}
              </div>
            ) : (
              <div className="camera-view">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted
                  onCanPlay={() => videoRef.current.play()}
                />
                
                <div className="scanner-overlay">
                  <div className="scan-area"></div>
                  {results.length > 0 && (
                    <div className="last-scan">
                      <p className="deck-name">{results[0].deckName}</p>
                      <p className="deck-price">${results[0].pricing.sellPrice.toFixed(2)}</p>
                    </div>
                  )}
                </div>
                
                <div className="camera-controls">
                  <button className="toggle-camera" onClick={toggleCamera}>
                    Flip Camera
                  </button>
                </div>
              </div>
            )}
            
            <div className="scanner-controls">
              {!scanning ? (
                <button 
                  className="start-scan-button" 
                  onClick={startScanning}
                  disabled={!cameraPermission}
                >
                  Start Scanning
                </button>
              ) : (
                <button className="stop-scan-button" onClick={stopScanning}>
                  Finish Scanning
                </button>
              )}
            </div>
          </>
        )}
      </main>
      
      {scanning && results.length > 0 && (
        <div className="results-preview">
          <h2>Scanned Decks</h2>
          <ul className="results-list">
            {results.map((result, index) => (
              <li key={index} className="result-item">
                <div className="result-name">{result.deckName}</div>
                <div className="result-price">${result.pricing.sellPrice.toFixed(2)}</div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ScannerView;

// ------------------------------------------------------
// src/components/SummaryView.jsx
// ------------------------------------------------------

import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

/**
 * Summary component - Displays scan results summary
 */
const SummaryView = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { summary } = location.state || { summary: null };
  const [exportFormat, setExportFormat] = useState('csv');
  
  // Handle missing summary
  if (!summary) {
    return (
      <div className="summary-error">
        <h1>No Data Available</h1>
        <p>No scanning data was found.</p>
        <button onClick={() => navigate('/scanner')}>
          Return to Scanner
        </button>
      </div>
    );
  }
  
  // Export data
  const handleExport = async () => {
    try {
      const scanner = new DeckScannerService();
      await scanner.initialize();
      
      // Set scan results
      scanner.scanResults = summary.results;
      
      // Export
      const blob = await scanner.exportResults(exportFormat);
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `deck-scan-results.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 100);
    } catch (error) {
      console.error('Export error:', error);
      alert('Failed to export results');
    }
  };
  
  return (
    <div className="summary-view">
      <header className="summary-header">
        <h1>Scan Summary</h1>
      </header>
      
      <main className="summary-main">
        <section className="summary-stats">
          <div className="stat-card total-decks">
            <h2>Total Decks</h2>
            <div className="stat-value">{summary.totalDecks}</div>
          </div>
          
          <div className="stat-card total-value">
            <h2>Total Value</h2>
            <div className="stat-value">${summary.totalSellValue}</div>
          </div>
          
          <div className="stat-card total-profit">
            <h2>Profit Potential</h2>
            <div className="stat-value">${summary.totalProfit}</div>
          </div>
          
          <div className="stat-card margin">
            <h2>Avg. Margin</h2>
            <div className="stat-value">{summary.averageMargin}</div>
          </div>
        </section>
        
        {summary.mostProfitable && (
          <section className="most-profitable">
            <h2>Most Profitable Deck</h2>
            <div className="profitable-card">
              <div className="card-name">{summary.mostProfitable.deckName}</div>
              <div className="card-details">
                <div className="buy-price">
                  Buy: ${summary.mostProfitable.pricing.buyPrice.toFixed(2)}
                </div>
                <div className="sell-price">
                  Sell: ${summary.mostProfitable.pricing.sellPrice.toFixed(2)}
                </div>
                <div className="profit">
                  Profit: ${(summary.mostProfitable.pricing.sellPrice - 
                          summary.mostProfitable.pricing.buyPrice).toFixed(2)}
                </div>
              </div>
            </div>
          </section>
        )}
        
        <section className="results-list">
          <h2>All Scanned Decks</h2>
          <table className="results-table">
            <thead>
              <tr>
                <th>Deck Name</th>
                <th>Buy Price</th>
                <th>Sell Price</th>
                <th>Profit</th>
                <th>Margin %</th>
              </tr>
            </thead>
            <tbody>
              {summary.results.map((result, index) => (
                <tr key={index}>
                  <td>{result.deckName}</td>
                  <td>${result.pricing.buyPrice.toFixed(2)}</td>
                  <td>${result.pricing.sellPrice.toFixed(2)}</td>
                  <td>${(result.pricing.sellPrice - 
                        result.pricing.buyPrice).toFixed(2)}</td>
                  <td>{((result.pricing.sellPrice - result.pricing.buyPrice) / 
                       result.pricing.buyPrice * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        
        <section className="export-section">
          <h2>Export Results</h2>
          <div className="export-options">
            <select 
              value={exportFormat} 
              onChange={e => setExportFormat(e.target.value)}
            >
              <option value="csv">CSV Format</option>
              <option value="json">JSON Format</option>
            </select>
            <button onClick={handleExport}>Export Data</button>
          </div>
        </section>
      </main>
      
      <footer className="summary-footer">
        <button onClick={() => navigate('/scanner')}>
          New Scan Session
        </button>
      </footer>
    </div>
  );
};

export default SummaryView;

// ------------------------------------------------------
// public/service-worker.js
// ------------------------------------------------------

const CACHE_NAME = 'deck-scanner-cache-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/static/js/main.chunk.js',
  '/static/js/0.chunk.js',
  '/static/js/bundle.js',
  '/manifest.json',
  '/models/model.json',
  '/models/weights.bin',
  '/models/labels.json',
  '/data/initial_data.json',
  '/tessdata/eng.traineddata'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.filter(cacheName => cacheName !== CACHE_NAME)
            .map(cacheName => caches.delete(cacheName))
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', event => {
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }
  
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          return response;
        }
        
        return fetch(event.request)
          .then(response => {
            // Cache valid responses
            if (response && response.status === 200 && response.type === 'basic') {
              const responseToCache = response.clone();
              caches.open(CACHE_NAME)
                .then(cache => cache.put(event.request, responseToCache));
            }
            
            return response;
          });
      })
      .catch(() => {
        // Fallback for offline HTML pages
        if (event.request.headers.get('accept').includes('text/html')) {
          return caches.match('/offline.html');
        }
      })
  );
});

// Handle background sync
self.addEventListener('sync', event => {
  if (event.tag === 'sync-data') {
    event.waitUntil(syncData());
  }
});

// Background sync implementation
const syncData = async () => {
  try {
    // Open database
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open('deck_scanner_db', 1);
      request.onerror = reject;
      request.onsuccess = event => resolve(event.target.result);
    });
    
    // Get pending sync tasks
    const tx = db.transaction('sync_queue', 'readonly');
    const store = tx.objectStore('sync_queue');
    const tasks = await new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onerror = reject;
      request.onsuccess = event => resolve(event.target.result);
    });
    
    // Process tasks
    for (const task of tasks) {
      // Implementation would connect to backend API
      console.log('Syncing task:', task);
    }
    
    // Clear processed tasks
    const clearTx = db.transaction('sync_queue', 'readwrite');
    const clearStore = clearTx.objectStore('sync_queue');
    await new Promise((resolve, reject) => {
      const request = clearStore.clear();
      request.onerror = reject;
      request.onsuccess = event => resolve();
    });
    
    return true;
  } catch (error) {
    console.error('Sync failed:', error);
    return false;
  }
};

// ------------------------------------------------------
// public/manifest.json
// ------------------------------------------------------

{
  "short_name": "Deck Scanner",
  "name": "Las Vegas Deck Scanner",
  "description": "Analyze and value Las Vegas playing cards for resale",
  "icons": [
    {
      "src": "icons/icon-72x72.png",
      "sizes": "72x72",
      "type": "image/png"
    },
    {
      "src": "icons/icon-96x96.png",
      "sizes": "96x96",
      "type": "image/png"
    },
    {
      "src": "icons/icon-128x128.png",
      "sizes": "128x128",
      "type": "image/png"
    },
    {
      "src": "icons/icon-144x144.png",
      "sizes": "144x144",
      "type": "image/png"
    },
    {
      "src": "icons/icon-152x152.png",
      "sizes": "152x152",
      "type": "image/png"
    },
    {
      "src": "icons/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "icons/icon-384x384.png",
      "sizes": "384x384",
      "type": "image/png"
    },
    {
      "src": "icons/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ],
  "start_url": "/",
  "display": "standalone",
  "theme_color": "#2c3e50",
  "background_color": "#ffffff",
  "orientation": "portrait",
  "categories": ["utilities", "business"],
  "shortcuts": [
    {
      "name": "Start Scanning",
      "short_name": "Scan",
      "description": "Begin a new deck scanning session",
      "url": "/scanner",
      "icons": [{ "src": "icons/scan-96x96.png", "sizes": "96x96" }]
    }
  ]
}

// ------------------------------------------------------
// netlify.toml
// ------------------------------------------------------

[build]
  publish = "build"
  command = "npm run build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[functions]
  directory = "functions"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
    Content-Security-Policy = "default-src 'self'; img-src 'self' data:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self' https://api.deckpricer.com;"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
    Permissions-Policy = "camera=self"
    
[build.environment]
  NODE_VERSION = "16"
  
[[plugins]]
  package = "@netlify/plugin-lighthouse"

  [plugins.inputs]
    output_path = "reports/lighthouse.html"
