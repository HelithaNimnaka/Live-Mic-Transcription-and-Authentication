# functions.py
import os
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

#Speaker Database
user_csv = "speakers/users.csv"


# Cache the model & processor at import time
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

def transcribe(audio_path: str) -> str:
    """
    Load an audio file and return Whisper’s transcription.
    """
    # 1. Load & resample
    speech, sr = librosa.load(audio_path, sr=16000)
    # 2. Feature-extract
    inputs = processor.feature_extractor(
        speech, sampling_rate=16000, return_tensors="pt"
    )
    attention_mask = torch.ones_like(inputs["input_features"]).long()
    # 3. Inference
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            attention_mask=attention_mask,
            language="en",
        )
    # 4. Decode and return
    return processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]


def manual_authentication(user_name, user_password, manual_csv=user_csv):
    """
    Authenticate a user based on a manual CSV file.
    """
    import pandas as pd
    df = pd.read_csv(manual_csv)

    if user_name in df['username'].values:
        user_row = df[df['username'] == user_name]
        stored_password = str(user_row['password'].values[0]).strip()
        input_password = user_password.strip()
        if stored_password == input_password:
            return "VERIFIED"
    return "REJECTED"



