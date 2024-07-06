from flask import Flask, request, jsonify, render_template
import tensorflow.keras.models as models
import librosa
import numpy as np
app = Flask(__name__)

# Load your trained model
model = models.load_model('your_model.h5')

# Define route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Function to extract features from audio files
def features_extractor(audio_file):
    audio, samplerate = librosa.load(audio_file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Define route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded audio file from the request
    audio_file = request.files['audio']
    # Save the uploaded audio file to a temporary location
    temp_audio_path = 'temp_audio.wav'
    audio_file.save(temp_audio_path)
    # Extract features from the uploaded audio file
    prediction_feature = features_extractor(temp_audio_path)
    prediction_feature = prediction_feature.reshape(1, -1)
    # Perform predictions using the model
    prediction = model.predict(prediction_feature)
    threshold = 0.5  # Adjust this threshold as needed
    if prediction[0][0] > threshold:
        predicted_label = "Positive Scream"
    else:
        predicted_label = "Negative Scream"
    
    # Render prediction result template with prediction
    return render_template('prediction_result.html', predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)

