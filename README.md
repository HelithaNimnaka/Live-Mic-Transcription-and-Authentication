# 🎤 Live Mic Transcription + Speaker Authentication

A Streamlit-based web application that enables live voice recording, transcription using Whisper, and speaker authentication using NVIDIA NeMo's speaker verification model. This project also supports manual login/enrollment for added user management.

---

## 🚀 Features

- 🎙️ **Live Voice Recording** (via microphone)
- 🧠 **Transcription** using [OpenAI Whisper](https://github.com/openai/whisper)
- 🔐 **Speaker Verification** using [NVIDIA NeMo Titanet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large)
- ✍️ **Manual Login / Enrollment** for traditional authentication
- 📁 **Speaker Embedding Database** for voice profile matching
- ✅ **User-friendly Interface** built with [Streamlit](https://streamlit.io)

---

## 🧠 Models Used

| Task                 | Model                               | Library         |
|----------------------|--------------------------------------|-----------------|
| **Transcription**    | `openai/whisper-tiny`                | 🤗 Transformers |
| **Speaker Verification** | `nvidia/speakerverification_en_titanet_large` | NVIDIA NeMo     |

---

## 🛠️ Tools & Libraries

- `Streamlit` – UI and interaction
- `SoundDevice` & `SoundFile` – Audio recording and saving
- `Transformers` – Whisper transcription
- `Librosa` – Audio processing
- `NVIDIA NeMo` – Speaker verification and embedding extraction
- `scikit-learn` – Cosine similarity
- `Pandas` – Manual user CSV management

---

## 📂 Project Structure
```
📁 project_root/
│
├── app.py # Main Streamlit app
├── functions.py # Transcription and manual auth logic
├── Speaker_Authontication.py # Speaker verification and embedding logic
├── speakers/
│ ├── users.csv # User credentials (username, password)
│ └── *.npy # Saved speaker embeddings
├── transcripts.txt # Transcription logs
└── README.md # This file
```
---

## 🧪 How It Works

1. **Record Voice**:
   - Click the "Start Recording" button.
   - The app records live audio from your mic.

2. **Transcription**:
   - Your voice is transcribed using OpenAI Whisper.

3. **Speaker Authentication**:
   - Extracts speaker embedding.
   - Compares with saved voice profiles using cosine similarity.
   - If matched, speaker is verified. Otherwise, user can enroll.

4. **Manual Auth (Fallback)**:
   - Users can log in or register using a simple form-based method.

---

## 🧰 Setup Instructions

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-auth-transcription.git
cd voice-auth-transcription

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 📦 Dependencies (requirements.txt)
```
Cython
nemo_toolkit[asr]
transformers==4.37.0
huggingface_hub==0.20.0
streamlit
sounddevice
```
Note: NeMo requires PyTorch and may need a compatible GPU for best performance.

## ▶️ Run the App
```
streamlit run app.py
```
## ✍️ Future Improvements

After obtaining the transcribed text, it can be passed to a large language model (LLM) to understand the intent and, using helper functions, generate relevant endpoint queries.

---
