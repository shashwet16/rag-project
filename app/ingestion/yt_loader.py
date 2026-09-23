from urllib.parse import urlparse, parse_qs
from app.ingestion.document import Document
# urlparse() → breaks a URL into parts like domain, path and query.
# parse_qs() → converts URL query parameters into a dictionary.
# Example: "?v=abc123" → {"v": ["abc123"]}

from youtube_transcript_api import YouTubeTranscriptApi
# YouTubeTranscriptApi → library used to retrieve YouTube captions/transcripts.
# We use it so our RAG system can ingest YouTube videos.


def extract_video_id(url):
    # Input → full YouTube URL.
    # Output → YouTube's video ID.
    # Example: https://www.youtube.com/watch?v=abc123 → abc123

    parsed = urlparse(url)
    # urlparse() separates the URL into components.
    # parsed.netloc → domain, e.g. "www.youtube.com"
    # parsed.path → path, e.g. "/embed/abc123"
    # parsed.query → query, e.g. "v=abc123"

    if "youtube.com" in parsed.netloc:
        # Check whether this is a normal youtube.com URL.

        query_params = parse_qs(parsed.query)
        # Convert the query string into a dictionary.
        # "v=abc123" → {"v": ["abc123"]}

        if "v" in query_params:
            # Check whether the URL contains the "v" parameter.
            # The "v" parameter contains the YouTube video ID.

            return query_params["v"][0]
            # parse_qs stores values inside lists.
            # {"v": ["abc123"]} → ["abc123"] → "abc123"

    if "youtu.be" in parsed.netloc:
        # Check for the shortened youtu.be format.
        # Example: https://youtu.be/abc123

        return parsed.path.lstrip("/")
        # parsed.path → "/abc123"
        # lstrip("/") → removes the leading "/" → "abc123"

    if "youtube.com" in parsed.netloc and "/embed/" in parsed.path:
        # Check for embedded YouTube URLs.
        # Example: https://www.youtube.com/embed/abc123

        return parsed.path.split("/embed/")[1]
        # split() separates the path around "/embed/".
        # [1] selects everything after "/embed/" → "abc123"

    raise ValueError(f"Could not extract video ID from: {url}")
    # raise ValueError → stop the function because we don't recognize the URL.


def load_youtube(url):
    # Input → YouTube URL.
    # Output → list containing one standardized RAG document.

    video_id = extract_video_id(url)
    # Convert the full URL into the video ID required by the transcript API.

    try:
        # try → run code that might fail.
        # Transcript fetching can fail if captions are unavailable, for example.

        api = YouTubeTranscriptApi()
        # Create an API client object.
        # The current library uses an instance instead of the old
        # YouTubeTranscriptApi.get_transcript() style.

        transcript = api.fetch(video_id).to_raw_data()
        # fetch(video_id) → retrieves the video's transcript.
        # to_raw_data() → converts the result into normal Python data.
        #
        # Result looks like:
        # [
        #     {"text": "Hello", "start": 0.0, "duration": 2.0},
        #     {"text": "everyone", "start": 2.0, "duration": 2.0}
        # ]

    except Exception as e:
        # except → runs if something inside try fails.
        # e → contains information about the error.

        print(f"Error fetching transcript: {e}")
        # Show the actual error so we know what went wrong.

        return []
        # Return an empty list because no document could be created.

    full_text = " ".join(
        segment["text"] for segment in transcript
    )
    # " ".join() → combines multiple strings using a space.
    # segment["text"] → gets the text from each transcript segment.
    #
    # ["Hello", "everyone", "welcome"]
    #              ↓
    # "Hello everyone welcome"
    #
    # We remove timestamps here because the basic RAG pipeline
    # only needs the actual text.

    if not full_text.strip():
        # strip() → removes whitespace from the beginning/end.
        # If nothing remains, the transcript is effectively empty.

        print(f"Warning: Empty transcript for {url}")
        # Tell us that extraction technically succeeded but produced no text.

        return []
        # No useful document → return an empty list.
    doc = Document(
        text=full_text,
        metadata={
            "source": url,
            "source_type": "youtube",
            "video_id": video_id,
        }
    )

    print(f"Extracted {len(full_text)} chars from video {video_id}")
    # len(full_text) → counts the extracted characters.
    # Useful for quickly checking whether ingestion worked.

    return [doc]
    # Return a list containing our document.
    # All ingestion loaders use lists so they have the same interface:
    #
    # PDF / URL / YouTube
    #        ↓
    #   list of documents
    #        ↓
    #     chunking