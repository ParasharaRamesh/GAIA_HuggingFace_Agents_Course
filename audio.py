import os

import whisper
from urllib.parse import urlparse, parse_qs
from langchain_core.tools import tool
from yt_dlp import YoutubeDL

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


@tool
def get_youtube_transcript(youtube_url: str) -> str:
    """
    Downloads the audio of a YouTube video and uses an existing method `transcribe_audio`
    to generate a transcript.

    Args:
        youtube_url (str): Full YouTube video URL.

    Returns:
        str: The transcript text or an error message.
    """
    try:
        # Extract video ID from URL
        parsed_url = urlparse(youtube_url)
        query = parse_qs(parsed_url.query)
        video_id = query.get("v")
        if not video_id:
            # Support for youtu.be short URL
            video_id = parsed_url.path.lstrip("/")
        else:
            video_id = video_id[0]

        # Download audio (MP3)
        audio_path = download_youtube_audio(youtube_url, video_id)

        # Call your existing transcription method (assumed to be defined elsewhere)
        transcript_text = transcribe_audio.invoke(audio_path)

        return transcript_text

    except Exception as e:
        return f"Error retrieving transcript: {e}"


def download_youtube_audio(youtube_url: str, video_id: str) -> str:
    """
    Downloads the audio of a YouTube video and saves it locally as MP3.

    Args:
        youtube_url (str): Full YouTube video URL.
        video_id (str): Video ID of the YouTube video.

    Returns:
        str: Path to the downloaded MP3 file.
    """
    audio_output_path = os.path.join(f"{video_id}.%(ext)s")


    ydl_opts_audio = {
        "format": "bestaudio/best",
        "outtmpl": audio_output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([youtube_url])

    return f"{video_id}.mp3"

if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=1htKBjuUWec"
    print(get_youtube_transcript(url))


