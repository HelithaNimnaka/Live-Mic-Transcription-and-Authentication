import tempfile
import os
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.utils.fetching import LocalStrategy

model = separator.from_hparams(
    source="speechbrain/sepformer-dns4-16k-enhancement",
    #source = "NVIDIA/speech_enhancement",
    savedir="pretrained_models/sepformer-dns4-16k-enhancement",
    local_strategy=LocalStrategy.COPY
)

#def voice_enhancement(filename):
#    try:
#        # Load original audio
#        waveform, original_sr = torchaudio.load(filename)
#
#        # Convert to mono if stereo
#        if waveform.shape[0] > 1:
#            waveform = waveform.mean(dim=0, keepdim=True)
#
#        # Resample to 16kHz for the model
#        if original_sr != 16000:
#            waveform = torchaudio.transforms.Resample(original_sr, 16000)(waveform)
#
#        # Prepare input shape: [batch, samples]
#        waveform = waveform.squeeze()
#        if waveform.dim() == 1:
#            waveform = waveform.unsqueeze(0)
#
#        # Run enhancement
#        est_sources = model.separate_batch(waveform)
#        clean = est_sources[0].detach().cpu()
#
#        # Prepare shape for saving: [channels, samples]
#        if clean.dim() == 1:
#            clean = clean.unsqueeze(0)
#        elif clean.dim() == 3:
#            clean = clean.squeeze(0)
#        elif clean.dim() == 2 and clean.shape[1] == 1 and clean.shape[0] > clean.shape[1]:
#            clean = clean.transpose(0, 1)
#
#        # Resample back to original sample rate (if needed)
#        if original_sr != 16000:
#            clean = torchaudio.transforms.Resample(16000, original_sr)(clean)
#
#        # Save enhanced audio
#        temp_file = tempfile.NamedTemporaryFile(suffix="_enhanced.wav", delete=False)
#        torchaudio.save(temp_file.name, clean, original_sr)
#        return temp_file.name
#
#    except Exception as e:
#        print(f"Enhancement failed: {e}")
#        #return filename


from clearvoice import ClearVoice
import soundfile as sf
# Initialize ClearVoice with the desired model
myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

def voice_enhancement(filename):
    """
    Enhance audio using ClearVoice and save to a temporary file.
    Returns the path to the enhanced audio file.
    """
    try:
        # Get enhanced audio as numpy array
        enhanced_audio = myClearVoice(input_path=filename, online_write=False)
        
        # Create a temporary file for the enhanced audio
        temp_file = tempfile.NamedTemporaryFile(suffix="_enhanced.wav", delete=False)
        temp_file.close()  # Close the file handle so we can write to it
        
        # Handle the (1, samples) format from ClearVoice
        if enhanced_audio.ndim == 2 and enhanced_audio.shape[0] == 1:
            # (1, samples) -> squeeze to 1D
            audio_to_save = enhanced_audio.squeeze()
        elif enhanced_audio.ndim == 2 and enhanced_audio.shape[1] == 1:
            # (samples, 1) -> squeeze to 1D
            audio_to_save = enhanced_audio.squeeze()
        elif enhanced_audio.ndim == 1:
            # Already 1D
            audio_to_save = enhanced_audio
        else:
            # Take first channel if multiple channels
            audio_to_save = enhanced_audio[0, :] if enhanced_audio.ndim == 2 else enhanced_audio

        # Ensure audio is float32 and in the right range
        if audio_to_save.dtype != 'float32':
            audio_to_save = audio_to_save.astype('float32')
        
        # Normalize if values are outside [-1, 1] range
        if audio_to_save.max() > 1.0 or audio_to_save.min() < -1.0:
            print("Normalizing audio to [-1, 1] range")
            audio_to_save = audio_to_save / max(abs(audio_to_save.max()), abs(audio_to_save.min()))
        
        # Save the enhanced audio to the temporary file
        # Use 48kHz since that's what ClearVoice typically outputs
        print(f"Saving to: {temp_file.name}")
        sf.write(temp_file.name, audio_to_save, 48000, subtype='PCM_16')
        
        return temp_file.name
        
    except Exception as e:
        print(f"Voice enhancement failed: {e}")
        # If enhancement fails, return the original filename
        #return filename


#process single wave file
#output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)
#myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')

#process wave directory
#myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

#process wave list file
#myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')