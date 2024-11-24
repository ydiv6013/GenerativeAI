# imports

import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Constants

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"


# Define our system prompt - you can experiment with this later, changing the last sentence to 'Respond in markdown in Spanish."

system_prompt = "Write a code snippets in python."

# static user promt

user_prompt = input('please write a code snippets requirement.....')
# Create a messages list using the same format that we used for OpenAI

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }


"""# call ollama using http 
response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
print(response.json()['message']['content'])"""

print("###################### Website Summary using ollama ##########################")
# call ollama using python ollama package
import ollama

response = ollama.chat(model=MODEL, messages=messages)

print(response['message']['content'])