# imports

import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Constants

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

# Waebsite page summary using ollama

class WebsiteScrap :
    """
        A utility class to represent a Website that we have scraped
    """
    def __init__(self,url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response  = requests.get(url=url)
        soup = BeautifulSoup(response.content,'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]) :
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)
        

web_url = WebsiteScrap(input("Please enter website to genereate page summary : \n"))
print(web_url.url)
print(web_url.title)
print(web_url.text)

# Define our system prompt - you can experiment with this later, changing the last sentence to 'Respond in markdown in Spanish."

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

# static user promt

user_prompt = web_url.text
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