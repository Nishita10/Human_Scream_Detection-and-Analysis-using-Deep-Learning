from flask import Flask, request, jsonify, render_template
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from twilio.rest import Client
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
ALERT_PHONE_NUMBER = os.getenv('ALERT_PHONE_NUMBER')

# Load pre-trained scream detection model
model = load_model('scream_detection_model(1).h5')

# Preprocess audio function
def preprocess_audio(audio_path, target_length=40, target_samplerate=16000):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Send SMS alert
def send_alert_message():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body="Scream detected! Immediate attention required.",
        from_=TWILIO_PHONE_NUMBER,
        to=ALERT_PHONE_NUMBER
    )
    return message.sid

# Route to serve the HTML UI
@app.route('/')
def index():
    return render_template('index.html')  # Ensure the HTML file is in a "templates" folder

# Route for scream detection
@app.route('/detect-scream', methods=['POST'])
def detect_scream():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_path = os.path.join("uploads", audio_file.filename)
    os.makedirs("uploads", exist_ok=True)
    audio_file.save(audio_path)

    try:
        # Preprocess the audio file
        processed_audio = preprocess_audio(audio_path)

        # Predict scream
        prediction = model.predict(processed_audio.reshape(1, -1))
        is_scream = prediction[0][0] < 0.5  # Adjust threshold if needed

        if is_scream:
            send_alert_message()
            result = {'result': 'Scream detected', 'message': 'Immediate action required!'}
        else:
            result = {'result': 'No scream detected', 'message': 'Everything seems calm.'}

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(audio_path)

if __name__ == '__main__':
    app.run(debug=True)
