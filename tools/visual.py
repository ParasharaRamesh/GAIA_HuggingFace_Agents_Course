import os
from yt_dlp import YoutubeDL

#TODO.x make this a tool?
#TODO.x any leight weight LLM which can do inference on videos/images directly so that I can make that into a seperate agent

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

if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=1htKBjuUWec"
    id = "1htKBjuUWec"
    print(download_youtube_video(url, id))
