# imports

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI


load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
api_key = os.getenv('OPENAI_API_KEY')

print(api_key)

# check the key
if not api_key : 
    print("No API key was found ")
else : 
    print("API key found ")


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

user_promt = web_url.text

"""
The API from OpenAI expects to receive messages in a particular structure. Many of the other APIs share this structure:

[
    {"role": "system", "content": "system message goes here"},
    {"role": "user", "content": "user message goes here"}
]
        
"""


# open AI
def web_summary(url):
    client =  OpenAI()
    data = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_promt}
        ]
    completion =  client.chat.completions.create(
        model="gpt-4o",
        messages= data,
    )
    return completion

summary = web_summary(web_url)
print(summary.choices[0].message.content)