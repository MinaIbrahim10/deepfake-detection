from tensorflow import keras
import tensorflow as tf
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel 
import uvicorn
from fastapi import FastAPI, File, UploadFile 
from io import BytesIO
import os
import pandas as pd
import librosa
model=joblib.load('deepfakevoiceusingML98%.h5')
app=FastAPI()
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)[:1]
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Stack features vertically
    features = np.vstack([chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfccs])

    # Take the mean along the feature axis
    features = np.mean(features, axis=1)
    print(chroma_stft.shape,rms.shape,spectral_bandwidth.shape,spectral_centroid.shape,rolloff.shape,zero_crossing_rate.shape,mfccs.shape)

    return features

async def index():
    return {"hola"}
    
@app.post("/predict")

async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("mp3", "wav",'rm')
    if not extension:
        return "File must be mp3 ,wav, or rm  format!"

    audio = await file.read()
    extracted_features = extract_features(BytesIO(audio))
    extracted_features = extracted_features.reshape(1, -1)
    columns=['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
       'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3',
       'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
       'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16',
       'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
    extracted_features_df = pd.DataFrame(extracted_features, columns=columns)
    prediction = model.predict(extracted_features_df)
    from sklearn.preprocessing import LabelEncoder
    lb=LabelEncoder()
    y1=['FAKE','REAL']
    lb.fit_transform(y1)
    lb.inverse_transform(prediction)
    predicted_class: str
    if lb.inverse_transform(prediction)=='FAKE':
       predicted_class ='machine generated'
    else:
       predicted_class =''
    result_dict = {'predicted_class': predicted_class}



   

    return result_dict
    @app.post('/')
    async def main(request: Request): 
        return await request.json()

