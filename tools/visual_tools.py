import base64
import os
from langchain_core.tools import tool

'''
# Can use V-JEPA2 for doing video analysis but not implementing it due to lack of GPU
from yt_dlp import YoutubeDL

def download_youtube_video(youtube_url: str, video_id: str) -> str:
    """
    Downloads the video of a YouTube video and saves it locally as MP4.

    Args:
        youtube_url (str): Full YouTube video URL.
        video_id (str): Video ID of the YouTube video.

    Returns:
        str: Filename of the downloaded MP4 file (e.g., "video_id.mp4").
    """
    video_output_path = os.path.join( f"{video_id}.%(ext)s")

    ydl_opts_video = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": video_output_path,
        "merge_output_format": "mp4",
        "quiet": True,
    }

    with YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([youtube_url])

    # Return only the filename
    return f"{video_id}.mp4"
'''

@tool
def read_image_and_encode(file_path: str) -> str:
    """
    Reads an image file from the specified local path, encodes it to Base64,
    and returns the Base64 string prefixed with the appropriate data URI.

    Args:
        file_path (str): The full path to the image file (e.g., 'image.png').

    Returns:
        str: A data URI string (e.g., 'data:image/png;base64,...')
             containing the Base64 encoded image, suitable for multimodal LLMs.
             Returns an error message if the file cannot be read or is not found.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    if not os.path.isfile(file_path):
        return f"Error: Not a file at {file_path}"

    try:
        # Determine MIME type based on extension (basic approach, could be more robust)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".png":
            mime_type = "image/png"
        elif ext == ".jpg" or ext == ".jpeg":
            mime_type = "image/jpeg"
        elif ext == ".gif":
            mime_type = "image/gif"
        else:
            return f"Error: Unsupported image format for {file_path}. Supported: .png, .jpg/.jpeg, .gif"

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        return f"Error reading or encoding image file {file_path}: {e}"
