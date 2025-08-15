import streamlit as st
import numpy as np
import pandas as pd
import librosa 
import soundfile as sf
import io

# Load dataset and configurations
dataset = pd.read_csv('dataset.csv')
num_mfcc = 100
num_mels = 128
num_chroma = 50

# Streamlit UI setup
st.set_page_config(page_title="Audio Deepfake Detection", layout="centered")

# Navigation
menu = ["Home", "Audio Detection"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("🏠 Welcome to the Audio Deepfake Detection App")
    st.write("Welcome to the Audio Deepfake Detection App. Navigate to the 'Audio Detection' tab to analyze an audio file.")

elif choice == "Audio Detection":
    st.title("🔍 Audio Deepfake Detection")
    st.write("Upload an audio file to check if it's real or deepfake.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Read the uploaded file as a buffer
        file_bytes = uploaded_file.read()
        file_stream = io.BytesIO(file_bytes)  # Convert to file-like object

        try:
            # Load audio using librosa from the BytesIO buffer
            X, sample_rate = librosa.load(file_stream, sr=None, mono=True)

            with st.spinner("Analyzing audio..."):
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
                mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
                chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
                zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
                flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)

                features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))

                # Find closest match
                distances = np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1)
                closest_match_idx = np.argmin(distances)
                closest_match_label = dataset.iloc[closest_match_idx, -1]
                total_distance = np.sum(distances)
                closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
                closest_match_prob_percentage = round(closest_match_prob * 100, 3)

            # Display results
            if closest_match_label == 'deepfake':
                st.error(f"🚨 Fake audio detected with {closest_match_prob_percentage}% certainty")
            else:
                st.success(f"✅ Real audio detected with {closest_match_prob_percentage}% certainty")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")