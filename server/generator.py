import json
import requests
import base64
from pathlib import Path

# config
DB_PATH = Path("layout_debug/questions.json")
OLLAMA_URL = "http://localhost:11434/api/chat"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_inference():
    """Test if Llama 3.2-Vision can see your first ingested question."""
    with open(DB_PATH, "r") as f:
        db = json.load(f)
    
    # grab question and related image
    q = db["ingestions"][-1]["questions"][3]
    img_b64 = encode_image(q["image_crops"][0])

    payload = {
        "model": "llama3.2-vision",
        "messages": [{
            "role": "user", 
            "content": f"Describe the logic or diagram in this question: {q['text']}",
            "images": [img_b64]
        }],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    print("AI Analysis of Seed Question:\n", response.json()['message']['content'])

if __name__ == "__main__":
    test_inference()
