"""
Configuration file for Smart Stress Detection System
"""

# Flask Configuration
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5001, 
    'SECRET_KEY': 'your-secret-key-here'  # Change this in production
}

# Model Configuration
MODEL_CONFIG = {
    'STRESS_MODEL_PATH': 'models/stress_model.pkl',
    'EMOTION_MODEL_PATH': 'models/best_emotion_model.pth',
    'EMOTION_LABELS': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'TARGET_SIZE': (48, 48),
    'NORMALIZE_MEAN': 0.5,
    'NORMALIZE_STD': 0.5
}

# Exercise Level Mapping
EXERCISE_MAPPING = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}

# Gender Mapping
GENDER_MAPPING = {
    'Male': 1,
    'Female': 0
}

# Stress Level Labels
STRESS_LEVELS = ['Low', 'Medium', 'High']

# UI Configuration
UI_CONFIG = {
    'MAX_AGE': 100,
    'MIN_AGE': 18,
    'MAX_SLEEP_HOURS': 20,
    'MIN_SLEEP_HOURS': 4,
    'MAX_WORK_HOURS': 80,
    'MIN_WORK_HOURS': 0
}
