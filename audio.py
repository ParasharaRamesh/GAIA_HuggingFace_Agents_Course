import whisper
import os
from langchain_core.tools import tool

# Load the base model
model = whisper.load_model("tiny")

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the audio file at the given file path using Whisper (base model).

    Args:
        file_path (str): Path to the MP3 or other audio file.

    Returns:
        str: Transcribed text.
    """
    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {e}"

if __name__ == '__main__':
    print("audio 1")
    audio1 = "1.mp3"
    print(transcribe_audio(audio1))
    print("--------------------")


    print("audio 2")
    audio1 = "2.mp3"
    print(transcribe_audio(audio1))
