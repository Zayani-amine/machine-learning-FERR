"""
Facial Expression Detection Web Application
Using ANN + HOG Features

Run with: python web_app.py
Open in browser: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
import os
import base64
from skimage.feature import hog

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
IMG_SIZE = 64

# Emotion info
EMOTION_INFO = {
    'angry': {'color': '#FF4444', 'emoji': '😠'},
    'disgust': {'color': '#9932CC', 'emoji': '🤢'},
    'fear': {'color': '#808080', 'emoji': '😨'},
    'happy': {'color': '#FFD700', 'emoji': '😊'},
    'neutral': {'color': '#4169E1', 'emoji': '😐'},
    'sad': {'color': '#1E90FF', 'emoji': '😢'},
    'surprise': {'color': '#FF6347', 'emoji': '😲'}
}


class EmotionDetector:
    """Core emotion detection engine using ANN + HOG."""
    
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.pca = None
        self.label_encoder = None
        self.face_cascade = None
        self.is_loaded = False
        
    def load_models(self):
        """Load the ANN model, PCA transformer, and label encoder."""
        try:
            ann_hog_path = os.path.join(self.models_dir, "system4_ann_hog.joblib")
            pca_hog_path = os.path.join(self.models_dir, "pca_hog.joblib")
            encoder_path = os.path.join(self.models_dir, "label_encoder.joblib")
            
            if not os.path.exists(ann_hog_path):
                raise FileNotFoundError(f"ANN model not found: {ann_hog_path}")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
            
            self.model = joblib.load(ann_hog_path)
            self.label_encoder = joblib.load(encoder_path)
            
            if os.path.exists(pca_hog_path):
                self.pca = joblib.load(pca_hog_path)
            
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            self.is_loaded = True
            return True, "Models loaded successfully!"
            
        except Exception as e:
            self.is_loaded = False
            return False, str(e)
    
    def predict_emotion(self, image):
        """Predict emotion from an image."""
        if not self.is_loaded:
            return None, None, None
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None, None
        
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_region = gray[y:y+h, x:x+w]
        
        # Preprocess
        resized = cv2.resize(face_region, (IMG_SIZE, IMG_SIZE))
        equalized = cv2.equalizeHist(resized)
        
        # Extract HOG
        features = hog(equalized, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        features = features.reshape(1, -1)
        
        if self.pca is not None:
            features = self.pca.transform(features)
        
        prediction = self.model.predict(features)[0]
        emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        return emotion, (x, y, w, h), equalized


# Initialize detector at module level (for gunicorn)
detector = EmotionDetector()
_load_success, _load_message = detector.load_models()
print(f"Model loading: {'✅ ' + _load_message if _load_success else '❌ ' + _load_message}")


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎭 Emotion Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: #888;
            font-size: 1.1rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        
        .image-section {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .image-container {
            position: relative;
            width: 100%;
            border-radius: 15px;
            overflow: hidden;
            background: #000;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        #uploadedImage {
            width: 100%;
            border-radius: 15px;
            display: none;
            max-height: 480px;
            object-fit: contain;
        }
        
        .placeholder {
            color: #666;
            text-align: center;
            padding: 40px;
        }
        
        .placeholder-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            color: #fff;
        }
        
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-section {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }
        
        .result-title {
            font-size: 1.2rem;
            color: #888;
            margin-bottom: 30px;
        }
        
        .emoji-display {
            font-size: 6rem;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .emotion-text {
            font-size: 2rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 20px;
        }
        
        .face-preview {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .face-preview h3 {
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 15px;
        }
        
        #facePreview {
            width: 100px;
            height: 100px;
            border-radius: 10px;
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .status {
            text-align: center;
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
            background: rgba(255,255,255,0.05);
        }
        
        .status.success { color: #4ade80; }
        .status.error { color: #f87171; }
        .status.info { color: #60a5fa; }
        
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
        }
        
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            cursor: pointer;
            width: 100%;
            height: 100%;
        }
        
        @media (max-width: 900px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .loader {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #3a7bd5;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎭 Facial Expression Detection</h1>
            <p class="subtitle">Powered by ANN + HOG Features</p>
        </header>
        
        <div class="main-content">
            <div class="image-section">
                <div class="image-container">
                    <div class="placeholder" id="placeholder">
                        <div class="placeholder-icon">📷</div>
                        <p>Upload an image or take a photo to detect emotions</p>
                    </div>
                    <img id="uploadedImage" src="" alt="Uploaded Image">
                </div>
                
                <div class="controls">
                    <div class="file-input-wrapper">
                        <button class="btn btn-secondary">
                            📁 Upload Image
                        </button>
                        <input type="file" id="imageUpload" accept="image/*">
                    </div>
                    <div class="file-input-wrapper">
                        <button class="btn btn-primary">
                            📸 Take Photo
                        </button>
                        <input type="file" id="cameraCapture" accept="image/*" capture="user">
                    </div>
                </div>
                
                <div id="status" class="status info">
                    Ready - Upload an image or take a photo
                </div>
            </div>
            
            <div class="result-section">
                <h2 class="result-title">Detection Result</h2>
                
                <div class="loader" id="loader"></div>
                
                <div id="resultContent">
                    <div class="emoji-display" id="emojiDisplay">❓</div>
                    <div class="emotion-text" id="emotionText" style="color: #888;">
                        No Detection
                    </div>
                </div>
                
                <div class="face-preview">
                    <h3>Detected Face</h3>
                    <img id="facePreview" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" alt="Face">
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadedImage = document.getElementById('uploadedImage');
        const placeholder = document.getElementById('placeholder');
        const imageUpload = document.getElementById('imageUpload');
        const cameraCapture = document.getElementById('cameraCapture');
        const status = document.getElementById('status');
        const emojiDisplay = document.getElementById('emojiDisplay');
        const emotionText = document.getElementById('emotionText');
        const facePreview = document.getElementById('facePreview');
        const loader = document.getElementById('loader');
        
        const emotionInfo = {
            'angry': { color: '#FF4444', emoji: '😠' },
            'disgust': { color: '#9932CC', emoji: '🤢' },
            'fear': { color: '#808080', emoji: '😨' },
            'happy': { color: '#FFD700', emoji: '😊' },
            'neutral': { color: '#4169E1', emoji: '😐' },
            'sad': { color: '#1E90FF', emoji: '😢' },
            'surprise': { color: '#FF6347', emoji: '😲' }
        };
        
        function updateResult(emotion, faceData) {
            if (emotion && emotionInfo[emotion]) {
                const info = emotionInfo[emotion];
                emojiDisplay.textContent = info.emoji;
                emotionText.textContent = emotion;
                emotionText.style.color = info.color;
            } else {
                emojiDisplay.textContent = '❌';
                emotionText.textContent = 'No Face';
                emotionText.style.color = '#888';
            }
            
            if (faceData) {
                facePreview.src = 'data:image/jpeg;base64,' + faceData;
            }
        }
        
        function setStatus(message, type) {
            status.textContent = message;
            status.className = 'status ' + type;
        }
        
        async function processImage(file) {
            loader.style.display = 'block';
            setStatus('Processing image...', 'info');
            
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                loader.style.display = 'none';
                
                if (data.success) {
                    placeholder.style.display = 'none';
                    uploadedImage.style.display = 'block';
                    uploadedImage.src = 'data:image/jpeg;base64,' + data.image;
                    updateResult(data.emotion, data.face);
                    setStatus('✅ Detected: ' + (data.emotion || 'No face').toUpperCase(), 'success');
                } else {
                    setStatus('❌ Error: ' + data.error, 'error');
                }
            } catch (error) {
                loader.style.display = 'none';
                setStatus('❌ Error processing image', 'error');
            }
        }
        
        imageUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            await processImage(file);
            e.target.value = '';
        });
        
        cameraCapture.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            await processImage(file);
            e.target.value = '';
        });
        
        // Check model status on load
        fetch('/status').then(r => r.json()).then(data => {
            if (!data.loaded) {
                setStatus('❌ Models not loaded: ' + data.message, 'error');
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/status')
def status():
    return jsonify({
        'loaded': detector.is_loaded if detector else False,
        'message': 'Models loaded' if (detector and detector.is_loaded) else 'Models not loaded'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    
    file = request.files['image']
    
    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'success': False, 'error': 'Could not read image'})
    
    # Predict
    emotion, bbox, preprocessed = detector.predict_emotion(image)
    
    # Draw on image
    if emotion:
        x, y, w, h = bbox
        info = EMOTION_INFO.get(emotion, {'color': '#00FF00'})
        color_hex = info['color'].lstrip('#')
        color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
        label = emotion.upper()
        font_scale = 1.0
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(image, (x, y-th-10), (x+tw+10, y), color, -1)
        cv2.putText(image, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2)
    
    # Encode images
    _, img_buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_buffer).decode('utf-8')
    
    face_base64 = None
    if preprocessed is not None:
        _, face_buffer = cv2.imencode('.jpg', preprocessed)
        face_base64 = base64.b64encode(face_buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'emotion': emotion,
        'image': img_base64,
        'face': face_base64
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🎭 Facial Expression Detection Web App")
    print("=" * 50)
    
    if detector.is_loaded:
        print(f"✅ Models loaded successfully!")
    else:
        print(f"❌ Models not loaded")
        print("   Make sure models are saved in 'saved_models' folder")
    
    print()
    print("🌐 Starting server...")
    print("   Open in browser: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
