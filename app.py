import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
import nltk
from nltk.corpus import cmudict
import librosa
import re
from flask import Flask, request, jsonify, send_from_directory, make_response, render_template
from flask_cors import CORS, cross_origin
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='.')
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5000", "http://localhost:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True
    }
})

# Create required directories if they don't exist
REQUIRED_DIRS = ['uploads', 'videos', 'audios/reference']
for directory in REQUIRED_DIRS:
    if not os.path.exists(directory):
        os.makedirs(directory)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download required NLTK resources (run once)
# nltk.download('cmudict')
# nltk.download('punkt')

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

def get_whisper_transcription(audio_path, model, processor):
    """Get transcription from Whisper model for an audio file."""
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Process audio
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
    return transcription.lower()

def text_to_arpabet(text):
    """Convert text to ARPAbet phoneme sequence."""
    # Load CMU pronouncing dictionary
    prondict = cmudict.dict()
    
    # Tokenize text
    words = nltk.word_tokenize(text.lower())
    
    # Convert words to ARPAbet
    phonemes = []
    for word in words:
        # Clean the word (remove punctuation)
        clean_word = re.sub(r'[^\w\s]', '', word)
        if not clean_word:
            continue
            
        # Get pronunciation from dictionary
        if clean_word in prondict:
            # Use first pronunciation variant
            word_phonemes = prondict[clean_word][0]
            # Remove stress markers (numbers)
            word_phonemes = [re.sub(r'\d+', '', p) for p in word_phonemes]
            phonemes.extend(word_phonemes)
        else:
            # For out-of-vocabulary words, keep as is but mark
            phonemes.append(f"<OOV:{clean_word}>")
    
    return phonemes

def normalized_wer(ref_phonemes, hyp_phonemes):
    """
    Calculate WER between reference and hypothesis phoneme sequences,
    handling different lengths properly.
    """
    # Convert phoneme lists to strings for jiwer
    ref_str = ' '.join(ref_phonemes)
    hyp_str = ' '.join(hyp_phonemes)
    
    # Calculate WER
    error_rate = wer(ref_str, hyp_str)
    
    return error_rate

def phoneme_similarity(ref_phonemes, hyp_phonemes):
    """
    Calculate phoneme-level similarity between reference and hypothesis.
    Returns a value between 0 (completely different) and 1 (identical).
    """
    # Create distance matrix for dynamic programming
    n, m = len(ref_phonemes), len(hyp_phonemes)
    distance = np.zeros((n + 1, m + 1))
    
    # Initialize first row and column
    for i in range(n + 1):
        distance[i, 0] = i
    for j in range(m + 1):
        distance[0, j] = j
    
    # Calculate Levenshtein distance
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_phonemes[i-1] == hyp_phonemes[j-1] else 1
            distance[i, j] = min(
                distance[i-1, j] + 1,           # deletion
                distance[i, j-1] + 1,           # insertion
                distance[i-1, j-1] + cost       # substitution
            )
    
    # Calculate similarity as 1 - normalized distance
    max_len = max(n, m)
    if max_len == 0:
        return 1.0  # Both sequences empty -> perfect match
    
    similarity = 1.0 - (distance[n, m] / max_len)
    return similarity

def compare_pronunciations(audio_path1, audio_path2):
    """Compare pronunciations between two audio files using Whisper and ARPAbet."""
    # Load Whisper model and processor
    
    
    # Get transcriptions
    trans1 = get_whisper_transcription(audio_path1, model, processor)
    trans2 = get_whisper_transcription(audio_path2, model, processor)
    
    # Convert to ARPAbet phonemes
    phonemes1 = text_to_arpabet(trans1)
    phonemes2 = text_to_arpabet(trans2)
    
    # Calculate metrics
    phoneme_wer = normalized_wer(phonemes1, phonemes2)
    phon_similarity = phoneme_similarity(phonemes1, phonemes2)
    
    return {
        "transcription1": trans1,
        "transcription2": trans2,
        "phonemes1": phonemes1,
        "phonemes2": phonemes2,
        "phoneme_wer": phoneme_wer,
        "phoneme_similarity": phon_similarity
    }

@app.route('/compare_pronunciation', methods=['POST', 'OPTIONS'])
@cross_origin(supports_credentials=True)
def handle_pronunciation_comparison():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
        
    try:
        print("Received pronunciation comparison request")
        print("Files:", request.files)
        print("Form:", request.form)
        
        if 'recorded_audio' not in request.files or 'reference_audio' not in request.form:
            print("Missing files:", request.files, request.form)
            return jsonify({'error': 'Missing audio files'}), 400

        # Save recorded audio
        recorded_audio = request.files['recorded_audio']
        recorded_filename = secure_filename('recorded_audio.wav')  # Use fixed filename
        recorded_path = os.path.join(app.config['UPLOAD_FOLDER'], recorded_filename)
        recorded_audio.save(recorded_path)
        print(f"Saved recorded audio to {recorded_path}")

        # Get reference audio path
        reference_filename = request.form['reference_audio'].split('/')[-1]  # Extract filename from path
        reference_path = os.path.join('audios', 'reference', reference_filename)
        print(f"Looking for reference audio at {reference_path}")

        if not os.path.exists(reference_path):
            print(f"Reference audio not found at {reference_path}")
            return jsonify({'error': f'Reference audio not found: {reference_path}'}), 404

        # Compare pronunciations
        print("Starting pronunciation comparison")
        results = compare_pronunciations(reference_path, recorded_path)
        print("Comparison complete:", results)

        # Clean up uploaded file
        os.remove(recorded_path)
        print("Cleaned up recorded audio file")

        response = jsonify({
            'phoneme_similarity': results['phoneme_similarity'],
            'phoneme_wer': results['phoneme_wer'],
            'transcription1': results['transcription1'],
            'transcription2': results['transcription2']
        })
        
        return response

    except Exception as e:
        print("Error in pronunciation comparison:", str(e))
        return jsonify({'error': str(e)}), 500

# Main route - serve welcome page
@app.route('/')
@cross_origin(supports_credentials=True)
def serve_welcome():
    return send_from_directory('.', 'welcome.html')

# Welcome page 2
@app.route('/welcome2')
@cross_origin(supports_credentials=True)
def serve_welcome2():
    return send_from_directory('.', 'welcome2.html')

# Lesson page
@app.route('/lesson')
@cross_origin(supports_credentials=True)
def serve_lesson():
    return send_from_directory('.', 'lesson.html')

# Learning page
@app.route('/learning')
@cross_origin(supports_credentials=True)
def serve_learning():
    return send_from_directory('.', 'learning.html')

@app.route('/submit_learning_plan', methods=['POST'])
@cross_origin(supports_credentials=True)
def submit_learning_plan():
    return send_from_directory('.', 'home.html')

# Home page
@app.route('/home')
@cross_origin(supports_credentials=True)
def serve_home():
    return send_from_directory('.', 'home.html')

# Search page
@app.route('/search')
@cross_origin(supports_credentials=True)
def serve_search():
    return send_from_directory('.', 'search.html')

# Dashboard page
@app.route('/dashboard')
@cross_origin(supports_credentials=True)
def serve_dashboard():
    return send_from_directory('.', 'dashboard.html')

# Profile page
@app.route('/profile')
@cross_origin(supports_credentials=True)
def serve_profile():
    return send_from_directory('.', 'profile.html')

@app.route('/calendar')
@cross_origin(supports_credentials=True)
def serve_calendar():
    return send_from_directory('.', 'calendar.html')

# Serve static files (like audio files)
@app.route('/audios/<path:filename>')
@cross_origin(supports_credentials=True)
def serve_audio(filename):
    response = send_from_directory('audios', filename)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Serve video files
@app.route('/videos/<path:filename>')
@cross_origin(supports_credentials=True)
def serve_video(filename):
    response = send_from_directory('videos', filename)
    response.headers['Content-Type'] = 'video/mp4'
    return response

if __name__ == "__main__":
    app.run(debug=True)

