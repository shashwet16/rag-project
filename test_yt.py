from app.ingestion.yt_loader import load_youtube

# Use any YouTube video that has captions
# This is a short Python tutorial — should have auto-captions
url = "https://www.youtube.com/watch?v=rfscVS0vtbw"

try:
    docs = load_youtube(url)

    if docs:
        doc = docs[0]
        print(f"Video ID: {doc.metadata['video_id']}")
        print(f"Source:   {doc.metadata['source_type']}")
        print(f"Chars:    {len(doc.text)}")
        print(f"\nFirst 300 chars:\n{doc.text[:300]}")
    else:
        print("No transcript available for this video")

except Exception as e:
    print(f"Error: {e}")