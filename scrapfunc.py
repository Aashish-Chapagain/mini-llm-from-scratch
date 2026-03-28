import requests
import json
import time
from bs4 import BeautifulSoup
import unicodedata

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip()


pages = [
    "https://conversationstartersworld.com/random-questions/",
    "https://conversationstartersworld.com/fun-questions-to-ask/",
    "https://conversationstartersworld.com/good-questions-to-ask/",
    "https://conversationstartersworld.com/this-or-that-questions/",
    "https://conversationstartersworld.com/questions-to-ask-a-guy/",
    "https://conversationstartersworld.com/questions-to-ask-a-girl/",
]

dataset = []

for url in pages:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    questions = soup.find_all(['h2', 'h3', 'strong'])

    for q in questions:
        question_text = clean_text(q.get_text(strip=True))
        answer_tag = q.find_next('p')
        answer_text = clean_text(answer_tag.get_text(strip=True)) if answer_tag else None
        

        if question_text and answer_text and question_text.endswith('?'):
            dataset.append({
                "input": question_text,
                "output": answer_text
            })

    
    time.sleep(2)

with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Done — {len(dataset)} total pairs saved") 


