import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown,display,update_display
from openai import OpenAI

load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
api_key = os.getenv('OPENAI_API_KEY')

print(api_key)

# check the key
if not api_key : 
    print("No API key was found ")
else : 
    print("API key found ")


# A class to represent a Webpage

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self,url):
        self.url = url
        response = requests.get(url=url)
        self.body = response.content
        soup = BeautifulSoup(self.body,'html.parser')
        self.title = soup.title.string if soup.title else 'no title found'
        if soup.body :
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator='\n',strip=True)
        else:
            self.text = " "
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
        
    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

"""web_url = input('Please enter web url to make brochure:')"""
web_url = "https://www.adbookee.com"
web= Website(web_url)
print(web.title)
print(web.url)
print(web.links)


# define system prompt
link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page Products or services page, or Careers/Jobs pages.\n"

link_system_prompt += "You should respond in JSON as in this example:"

link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""


print(link_system_prompt)

# define user_prompt
def get_user_prompt(web_url):
    user_prompt = f"Here is the list of links on the website of {web.url} -"
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
    Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(web.links)
    return user_prompt

user_prompt = get_user_prompt(web_url)
print(user_prompt)

# extract the useful links using openn ai
def get_links(url):
    web = Website(url=url)
    client = OpenAI()
    data = [
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    response = client.chat.completions.create(
        model= 'gpt-4o-mini',
        messages= data,
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

link_json = get_links(web_url)

print(link_json)

# get the information from the url and make brochure

def get_info(url):
    result = 'Landing Page \n'
    result += Website(url).get_contents()
    links = get_links(url)
    print('Found Links',links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

page_info = get_info(web_url)

system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture,products,services, customers and careers/jobs if you have the information."

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_info(url)
    user_prompt = user_prompt[:20_000] # Truncate if more than 20,000 characters
    return user_prompt


# extract the brochure using openn ai
def create_brochure(url):
    web = Website(url=url)
    client = OpenAI()
    data = [
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt("Adbookee", url)}
        ]
    response = client.chat.completions.create(
        model= 'gpt-4o-mini',
        messages= data,
    )
    result = response.choices[0].message.content
    return result
    
brochure = create_brochure(web_url)
print(brochure)

