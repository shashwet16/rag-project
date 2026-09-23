import requests
from bs4 import BeautifulSoup 
from urllib.parse import urlparse # url parse breas
from app.ingestion.document import Document

def load_url(url):

    #validate url 
    parsed = urlparse(url)

    if parsed.scheme not in ('https' , 'https'):
        raise ValueError (f"invalid URL(must conatian http or https) - {url}")
    # fetch the apge 
    headers = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
        } 

    response = requests.get(
    url,
    headers=headers,
    timeout=10
    )

    response.raise_for_status()

    #parse the htmml 
     
    soup = BeautifulSoup(response.text , 'html.parser')

    # remove the uncessary elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
     tag.decompose()
    #eztract 
    text = soup.get_text(separator="\n")
    # --- Clean up whitespace ---
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    
    if not text.strip():
        print(f"  Warning: No text extracted from {url}")
        return []
    
    # --- Get title ---
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    
    doc = Document(
        text=text,
        metadata={
            "source": url,
            "source_type": "url",
            "title": title,
            "domain": parsed.netloc,
        }
    )
    print(f"  Extracted {len(text)} chars from {parsed.netloc}")
    return [doc]

