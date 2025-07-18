import nemo.collections.asr as nemo_asr
import librosa
import torch
import numpy as np
import soundfile as sf
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load the speaker verification model
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

speaker_db = "speakers"


def extract_embedding(audio_path):
    """
    Extract speaker embedding from audio file
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        torch.Tensor: Speaker embedding vector
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Convert to WAV format and extract embedding
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        temp_wav_path = f"temp_audio_{hash(audio_path) % 1000}.wav"
        sf.write(temp_wav_path, audio, sr)
        
        embedding = speaker_model.get_embedding(temp_wav_path)
        
        # Clean up temporary file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            
        return embedding
        
    except Exception as e:
        # Fallback method: Direct forward pass
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Ensure minimum length (1 second)
        min_samples = 16000
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
        
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
        
        speaker_model.eval()
        with torch.no_grad():
            logits, embedding = speaker_model.forward(input_signal=audio_tensor, input_signal_length=audio_length)
        
        return embedding

def verify_speakers(audio_path, speaker_db="speakers", threshold=0.7):
    """
    Verify input audio against a database of speaker embeddings.

    Args:
        audio_path (str): Path to input audio file
        speaker_db (str): Directory containing saved speaker embeddings (.npy or .pt)
        threshold (float): Similarity threshold

    Returns:
        dict: Contains matched speaker name (or 'REJECTED'), similarity score, and threshold
    """
    import glob

    # Extract embedding for the input audio
    input_emb = extract_embedding(audio_path)
    input_emb_np = input_emb.detach().cpu().numpy().reshape(1, -1)

    best_match = None
    best_score = -1

    # Loop through all saved embeddings in speaker_db
    for emb_path in glob.glob(os.path.join(speaker_db, "*.npy")):
        speaker_name = os.path.splitext(os.path.basename(emb_path))[0]
        db_emb = np.load(emb_path).reshape(1, -1)

        # Cosine similarity
        sim = cosine_similarity(input_emb_np, db_emb)[0][0]

        if sim > best_score:
            best_score = sim
            best_match = speaker_name

    if best_score >= threshold:
        return {
            "match": best_match,
            "similarity": best_score,
            "threshold": threshold,
            "status": "VERIFIED"
        }
    else:
        return {
            "match": None,
            "similarity": best_score,
            "threshold": threshold,
            "status": "REJECTED"
        }


#def saving_speaker_embedding(name, audio_path, speaker_db="speakers"):
#    """
#    Extract and save speaker embedding to the database.
#    """
#    os.makedirs(speaker_db, exist_ok=True)
#    embedding = extract_embedding(audio_path)
#    embedding_np = embedding.detach().cpu().numpy()
#    np.save(os.path.join(speaker_db, f"{name}.npy"), embedding_np)
#