"""
tts_utils.py — PharmaAgent Sesli Asistan Modülü
Groq Whisper STT + gtts TTS
"""
import io
import os
import base64
import tempfile

def transcribe_audio(audio_bytes: bytes, groq_client) -> str:
    """Groq Whisper ile sesi metne çevirir."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcript = groq_client.audio.transcriptions.create(
                file=("audio.wav", f),
                model="whisper-large-v3-turbo",
                language="tr",
                response_format="text"
            )
        os.unlink(tmp_path)
        return str(transcript).strip()
    except Exception as e:
        return f"Ses tanıma hatası: {str(e)}"


def text_to_speech(text: str) -> bytes:
    """gTTS ile metni Türkçe sese çevirir, bytes döner."""
    try:
        from gtts import gTTS
        clean_text = text[:1000]  # API limiti için kısalt
        tts = gTTS(text=clean_text, lang='tr', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        print(f"TTS Hatası: {e}")
        return b""


def get_audio_html(audio_bytes: bytes) -> str:
    """Ses bytes'ını Streamlit HTML audio player'a çevirir (autoplay)."""
    if not audio_bytes:
        return ""
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio autoplay controls style="width:100%;margin-top:8px;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
