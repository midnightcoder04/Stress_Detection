from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import base64
import io
from PIL import Image
import os
from config import *

# Optional imports with proper module-level scope
joblib = None
cv2 = None
torch = None
nn = None
transforms = None

try:
    import joblib
except ImportError:
    print("Warning: joblib not found. Stress prediction will use fallback method.")
    joblib = None

try:
    import cv2
except ImportError:
    print("Warning: opencv-python not found. Some image processing features may be limited.")
    cv2 = None

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
except ImportError:
    print("Warning: PyTorch not found. Emotion detection will use fallback method.")
    torch = None
    nn = None
    transforms = None

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not found. Stress prediction may not work properly.")
    pd = None
    

app = Flask(__name__)
app.secret_key = FLASK_CONFIG['SECRET_KEY']

# Load models
stress_model = None
emotion_model = None

def load_models():
    global stress_model, emotion_model
    
    # Load stress prediction model
    try:
        if joblib is not None:
            model_path = MODEL_CONFIG['STRESS_MODEL_PATH']
            print(f"[INIT] Attempting to load stress model from: {model_path}")
            
            # Check if file exists
            if not os.path.exists(model_path):
                print(f"[ERROR] Stress model file not found at: {model_path}")
                print(f"[INFO] Current working directory: {os.getcwd()}")
                print(f"[INFO] Files in models/ directory:")
                if os.path.exists('models'):
                    for f in os.listdir('models'):
                        print(f"  - {f}")
                else:
                    print("  models/ directory does not exist!")
            else:
                stress_model = joblib.load(model_path)
                print(f"[SUCCESS] Stress model loaded successfully")
                print(f"[DEBUG] Model type: {type(stress_model)}")
        else:
            print("[ERROR] Joblib not available - cannot load stress model")
    except Exception as e:
        print(f"[ERROR] Failed to load stress model: {e}")
        import traceback
        traceback.print_exc()
    
    # Load emotion detection model  
    print(f"[INIT] Checking PyTorch availability: torch={torch is not None}, nn={nn is not None}")
    if torch is not None and nn is not None:
        print(f"[INIT] PyTorch is available. Loading emotion model from: {MODEL_CONFIG['EMOTION_MODEL_PATH']}")
        try:
            print(f"[INIT] Checking if model file exists...")
            if not os.path.exists(MODEL_CONFIG['EMOTION_MODEL_PATH']):
                print(f"[ERROR] Model file not found at: {MODEL_CONFIG['EMOTION_MODEL_PATH']}")
                emotion_model = None
            else:
                print(f"[INIT] Model file exists. Loading...")
                loaded_object = torch.load(MODEL_CONFIG['EMOTION_MODEL_PATH'], map_location=torch.device('cpu'), weights_only=False)
                print(f"[INIT] Loaded object type: {type(loaded_object)}")
                
                # Handle different model formats
                if hasattr(loaded_object, 'eval'):
                    # It's a complete model
                    emotion_model = loaded_object
                    emotion_model.eval()
                    print("[SUCCESS] Complete model loaded and set to eval mode")
                elif isinstance(loaded_object, dict):
                    # It's a state_dict - need to create model architecture
                    print("[INIT] Detected state_dict format. Creating EmotionCNN architecture...")
                    
                    # Define the exact EmotionCNN architecture
                    try:
                        import torch.nn.functional as F
                        
                        class EmotionCNN(nn.Module):
                            def __init__(self, num_classes=7):
                                super(EmotionCNN, self).__init__()
                                # Enhanced convolutional layers with more filters
                                # Match the saved checkpoint: 1 input channel (grayscale), 3x3 kernel, padding=1
                                self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                                self.bn1 = nn.BatchNorm2d(64)
                                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                                self.bn2 = nn.BatchNorm2d(128)
                                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                                self.bn3 = nn.BatchNorm2d(256)
                                self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
                                self.bn4 = nn.BatchNorm2d(512)
                                self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
                                self.bn5 = nn.BatchNorm2d(512)

                                self.pool = nn.MaxPool2d(2, 2)
                                self.dropout = nn.Dropout(0.5)

                                # Calculate the size of the flattened features
                                self._to_linear = None
                                # Use 1 channel (grayscale) as in the checkpoint
                                self._get_conv_output((1, 1, 48, 48))

                                # Enhanced fully connected layers with more neurons
                                self.fc1 = nn.Linear(self._to_linear, 2048)
                                self.fc2 = nn.Linear(2048, 1024)
                                self.fc3 = nn.Linear(1024, 512)
                                self.fc4 = nn.Linear(512, 256)
                                self.fc5 = nn.Linear(256, num_classes)

                            def _get_conv_output(self, shape):
                                input = torch.rand(shape)
                                output = self._forward_conv(input)
                                self._to_linear = int(np.prod(output.shape))

                            def _forward_conv(self, x):
                                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                                x = self.pool(F.relu(self.bn4(self.conv4(x))))
                                x = self.pool(F.relu(self.bn5(self.conv5(x))))
                                return x

                            def forward(self, x):
                                x = self._forward_conv(x)
                                x = x.view(-1, self._to_linear)
                                x = F.relu(self.fc1(x))
                                x = self.dropout(x)
                                x = F.relu(self.fc2(x))
                                x = self.dropout(x)
                                x = F.relu(self.fc3(x))
                                x = self.dropout(x)
                                x = F.relu(self.fc4(x))
                                x = self.dropout(x)
                                x = self.fc5(x)
                                return x
                        
                        # Create model instance
                        num_emotions = len(MODEL_CONFIG['EMOTION_LABELS'])
                        emotion_model = EmotionCNN(num_classes=num_emotions)
                        
                        # Load the state dict
                        emotion_model.load_state_dict(loaded_object)
                        emotion_model.eval()
                        print(f"[SUCCESS] EmotionCNN architecture created and state_dict loaded successfully for {num_emotions} emotions")
                    except Exception as arch_error:
                        print(f"[ERROR] Failed to create model architecture: {arch_error}")
                        import traceback
                        traceback.print_exc()
                        emotion_model = None
                else:
                    print(f"[ERROR] Unknown model format: {type(loaded_object)}")
                    emotion_model = None
        except Exception as e:
            print(f"[ERROR] Error loading emotion model: {e}")
            import traceback
            traceback.print_exc()
            emotion_model = None
    else:
        print("[ERROR] PyTorch not available - INSTALL IT: pip install torch torchvision")
        print("[ERROR] Without PyTorch, emotion detection will not work properly!")
        emotion_model = None

# Emotion labels
EMOTION_LABELS = MODEL_CONFIG['EMOTION_LABELS']

# Image preprocessing for emotion detection
def preprocess_image(image_data):
    """Convert base64 image to grayscale 48x48 tensor"""
    if transforms is None:
        print("PyTorch transforms not available")
        return None
        
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to target size
        image = image.resize(IMAGE_CONFIG['TARGET_SIZE'])
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMAGE_CONFIG['NORMALIZE_MEAN']], std=[IMAGE_CONFIG['NORMALIZE_STD']])
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion/<path:filename>')
def serve_emotion(filename):
    """Serve emotion sample images"""
    emotion_dir = os.path.join(app.root_path, 'emotion')
    return send_from_directory(emotion_dir, filename)

@app.route('/emotion/list')
def list_emotions():
    """List all emotion sample images"""
    emotion_dir = os.path.join(app.root_path, 'emotion')
    try:
        files = []
        if os.path.isdir(emotion_dir):
            for name in os.listdir(emotion_dir):
                lower = name.lower()
                if lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                    files.append(name)
        return jsonify({'images': sorted(files)})
    except Exception as e:
        return jsonify({'images': [], 'error': str(e)}), 500

@app.route('/predict_stress', methods=['POST'])
def predict_stress():
    try:
        data = request.get_json()
        
        # Extract features
        age = int(data['age'])
        gender = GENDER_MAPPING[data['gender']]
        exercise_level = data['exercise_level']
        sleep_hours = float(data['sleep_hours'])
        work_hours = float(data['work_hours'])
        
        # Convert exercise level to numeric
        exercise_numeric = EXERCISE_MAPPING[exercise_level]
        
        # REQUIRE ML model to be loaded - no fallback
        if stress_model is None:
            error_msg = "Stress prediction model not loaded. Please ensure scikit-learn and joblib are installed and the model file exists at " + MODEL_CONFIG['STRESS_MODEL_PATH']
            print(f"[ERROR] {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Check if pandas is available (required for model input)
        if pd is None:
            error_msg = "Pandas is required for stress prediction. Install it with: pip install pandas"
            print(f"[ERROR] {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Create feature DataFrame for ML model (model expects specific column names)
        # IMPORTANT: Model was trained with categorical string labels, not numeric codes!
        # Gender and Exercise Level must be strings (they go through OneHotEncoder)
        features_dict = {
            'Age': [int(age)],
            'Gender': [data['gender']],  # Use original string: 'Male' or 'Female'
            'Exercise Level': [exercise_level],  # Use original string: 'Low', 'Medium', 'High'
            'Sleep Hours': [float(sleep_hours)],
            'Work Hours per Week': [float(work_hours)]
        }
        features = pd.DataFrame(features_dict)
        
        print(f"[DEBUG] Predicting stress with features: age={age}, gender={gender}, exercise={exercise_numeric}, sleep={sleep_hours}, work={work_hours}")
        print(f"[DEBUG] DataFrame shape: {features.shape}, columns: {list(features.columns)}")
        print(f"[DEBUG] DataFrame dtypes: {features.dtypes.to_dict()}")
        
        # Get ML model prediction (supports both regression and classification models)
        try:
            ml_prediction = stress_model.predict(features)[0]
            
            # Log the model's prediction for debugging
            print(f"[DEBUG] Model prediction output: {ml_prediction}")

            # Check if the model outputs a numeric value (regression) or string (classification)
            if isinstance(ml_prediction, (int, float, np.number)):
                # Regression model - continuous stress score (0-100)
                stress_score = float(np.clip(ml_prediction, 0, 100))
                
                # Convert continuous score to categorical level for display
                if stress_score < 33.3:
                    stress_level = "Low"
                elif stress_score < 66.6:
                    stress_level = "Medium"
                else:
                    stress_level = "High"
                
                print(f"[DEBUG] Stress score: {stress_score:.2f}, Level: {stress_level}")
                
                return jsonify({
                    "stress_level": stress_level,
                    "stress_score": round(stress_score, 2)
                })
            else:
                # Classification model - categorical output (Low/Medium/High/Medium)
                stress_level = str(ml_prediction)
                
                # Normalize the output (handle 'Medium' -> 'Medium')
                if stress_level.lower() == 'Medium':
                    stress_level = "Medium"
                
                # Generate an approximate score based on the level for backward compatibility
                score_mapping = {"Low": 20, "Medium": 50, "High": 80}
                stress_score = score_mapping.get(stress_level, 50)
                
                print(f"[DEBUG] Classification model output - Level: {stress_level}, Estimated score: {stress_score}")
                
                return jsonify({
                    "stress_level": stress_level,
                    "stress_score": stress_score
                })
        except Exception as model_error:
            print(f"[ERROR] ML model prediction failed: {model_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Model prediction failed: {str(model_error)}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess image
        tensor = preprocess_image(image_data)
        if tensor is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Predict emotions - REQUIRE model to be loaded
        if emotion_model is None:
            error_msg = "Emotion model not loaded. Please ensure PyTorch is installed (pip install torch torchvision) and the model file exists at " + MODEL_CONFIG['EMOTION_MODEL_PATH']
            print(f"[ERROR] {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Use the actual model - no fallback to random!
        if torch is None:
            error_msg = "PyTorch is not available. Install it with: pip install torch torchvision"
            print(f"[ERROR] {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        with torch.no_grad():
            outputs = emotion_model(tensor)
            print(f"[DEBUG] Model raw outputs shape: {outputs.shape}")
            print(f"[DEBUG] Model raw outputs: {outputs}")
            
            probabilities = torch.softmax(outputs, dim=1)
            print(f"[DEBUG] Softmax probabilities shape: {probabilities.shape}")
            print(f"[DEBUG] Softmax probabilities: {probabilities}")
            
            emotion_probs = probabilities[0].numpy()
            print(f"[DEBUG] Numpy probabilities: {emotion_probs}")
        
        # Create emotion results
        emotion_results = {}
        for i, label in enumerate(EMOTION_LABELS):
            emotion_results[label] = float(emotion_probs[i] * 100)
        
        print(f"[DEBUG] Emotion labels: {EMOTION_LABELS}")
        print(f"[DEBUG] Emotion results (percentages): {emotion_results}")
        
        dominant_emotion = EMOTION_LABELS[np.argmax(emotion_probs)]
        print(f"[DEBUG] Dominant emotion: {dominant_emotion} ({emotion_results[dominant_emotion]:.2f}%)")
        
        return jsonify({
            'emotions': emotion_results,
            'dominant_emotion': dominant_emotion
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/combined_analysis', methods=['POST'])
def combined_analysis():
    """
    Combine lifestyle stress factors with emotional state to give a comprehensive stress score.
    Emotions like Angry, Fear, Sad, Disgust increase stress.
    Emotions like Happy, Surprise, Neutral decrease stress.
    """
    try:
        data = request.get_json()
        stress_data = data['stress_data']
        emotion_data = data['emotion_data']
        
        # Get base lifestyle stress score (0-100)
        # Support both 'stress_score' and 'combined_score' for backward compatibility
        lifestyle_score = stress_data.get('stress_score') or stress_data.get('combined_score', 50)
        
        # Get dominant emotion and percentages
        dominant_emotion = emotion_data['dominant_emotion']
        emotions = emotion_data['emotions']
        
        # Define emotion impact on stress
        # Negative emotions increase stress, positive emotions decrease it
        emotion_stress_weights = {
            'Angry': 15,      # High stress
            'Fear': 12,       # High stress
            'Sad': 10,        # Medium-high stress
            'Disgust': 8,     # Medium stress
            'Surprise': -3,   # Slight stress reduction (could be positive or negative)
            'Neutral': -5,    # Mild stress reduction
            'Happy': -10      # Significant stress reduction
        }
        
        # Calculate emotion-based adjustment
        # Weight by the percentage of each emotion
        emotion_adjustment = 0
        for emotion, percentage in emotions.items():
            weight = emotion_stress_weights.get(emotion, 0)
            # Scale by percentage (0-100) and normalize
            emotion_adjustment += (weight * percentage / 100)
        
        # Round to nearest integer
        emotion_adjustment = int(round(emotion_adjustment))
        
        # Calculate final combined stress score
        final_stress_score = lifestyle_score + emotion_adjustment
        
        # Clamp to 0-100 range
        final_stress_score = max(0, min(100, final_stress_score))
        
        # Determine final stress level
        if final_stress_score <= 35:
            final_stress_level = "Low"
        elif final_stress_score <= 60:
            final_stress_level = "Medium"
        else:
            final_stress_level = "High"
        
        # Generate overall assessment
        if final_stress_level == "Low":
            overall_assessment = f"Excellent! Your lifestyle factors and emotional state indicate low stress levels. Your {dominant_emotion.lower()} emotional state is working in your favor. Keep maintaining this balance!"
        elif final_stress_level == "Medium":
            if emotion_adjustment > 0:
                overall_assessment = f"You're experiencing Medium stress. While your lifestyle factors are manageable, your emotional state ({dominant_emotion}) is contributing to increased stress. Consider stress management techniques like meditation or talking to someone."
            else:
                overall_assessment = f"Medium stress detected, but your positive emotional state ({dominant_emotion}) is helping! Focus on improving sleep, exercise, and work-life balance to reduce stress further."
        else:
            if emotion_adjustment > 0:
                overall_assessment = f"High stress alert! Both your lifestyle factors and emotional state ({dominant_emotion}) indicate significant stress. Immediate action recommended: prioritize rest, seek support, and consider professional help if needed."
            else:
                overall_assessment = f"High stress detected from lifestyle factors, though your {dominant_emotion.lower()} emotional state is providing some relief. Focus urgently on improving sleep, reducing work hours, and increasing exercise."
        
        # Generate impact descriptions
        lifestyle_impact = f"Your age, exercise, sleep, and work habits contribute {lifestyle_score} points to your stress score."
        
        if emotion_adjustment > 0:
            emotion_impact = f"Your {dominant_emotion.lower()} emotional state adds {emotion_adjustment} stress points. Negative emotions like {dominant_emotion} can amplify stress."
        elif emotion_adjustment < 0:
            emotion_impact = f"Your {dominant_emotion.lower()} emotional state reduces stress by {abs(emotion_adjustment)} points. Positive emotions help manage stress!"
        else:
            emotion_impact = f"Your {dominant_emotion.lower()} emotional state has a neutral effect on your stress level."
        
        # Generate personalized recommendations
        recommendations = []
        
        # Based on lifestyle factors (if detailed_scores available)
        detailed_scores = stress_data.get('detailed_scores', {})
        
        if detailed_scores.get('sleep', {}).get('score', 0) > 20:
            recommendations.append("Prioritize sleep: Aim for 7-8 hours per night to reduce stress significantly.")
        
        if detailed_scores.get('exercise', {}).get('score', 0) > 15:
            recommendations.append("Increase physical activity: Regular exercise is proven to reduce stress hormones.")
        
        if detailed_scores.get('work_hours', {}).get('score', 0) > 15:
            recommendations.append("Reduce work hours: Consider setting boundaries to achieve better work-life balance.")
        
        # Based on emotions
        if dominant_emotion in ['Angry', 'Fear', 'Sad']:
            recommendations.append(f"Manage {dominant_emotion.lower()} feelings: Try deep breathing exercises, journaling, or speaking with a counselor.")
        
        if emotion_adjustment > 5:
            recommendations.append("Practice mindfulness: Daily meditation can help regulate negative emotions and reduce stress.")
        
        if dominant_emotion == 'Happy':
            recommendations.append("Maintain your positive outlook: Your happiness is a powerful stress buffer. Keep doing what makes you happy!")
        
        # General recommendations
        if final_stress_score > 60:
            recommendations.append("Consider professional support: High stress levels benefit from professional guidance.")
        
        if not recommendations:
            recommendations.append("Keep up the great work! Your stress levels are well-managed.")
        
        return jsonify({
            'final_stress_score': final_stress_score,
            'final_stress_level': final_stress_level,
            'lifestyle_score': lifestyle_score,
            'emotion_adjustment': emotion_adjustment,
            'dominant_emotion': dominant_emotion,
            'overall_assessment': overall_assessment,
            'lifestyle_impact': lifestyle_impact,
            'emotion_impact': emotion_impact,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error in combined analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(
        debug=FLASK_CONFIG['DEBUG'],
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT']
    )
