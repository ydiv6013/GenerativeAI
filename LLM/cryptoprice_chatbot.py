import requests
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr  

# define constants
load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()
MODEL = "gpt-4o-mini"
    
# check the key
if not api_key : 
    print("No API key was found ")
else : 
    print("API key found ")
    

system_prompt = "You are a knowledgeable and helpful cryptocurrency assistant"
system_prompt += "Your primary role is to provide real-time cryptocurrency prices, explain market trends, and assist users with cryptocurrency-related inquiries in a clear and concise manner."
system_prompt +="You can handle questions about specific coins, market analysis, trading concepts, and general cryptocurrency information. Always ensure your responses are accurate, user-friendly, and engaging."
system_prompt +="If a user asks for real-time prices or data, remind them that external sources may be required to fetch the most up-to-date information."

print(system_prompt)
"""
# chat function
def chat(message,history):
    message = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    
    stream  = openai.chat.completions.create(
        model=MODEL,
        messages=message,
        stream= True,
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ''
        yield result
        
# use gradio chatinterface for ui
gr.ChatInterface(
    fn=chat,
    type="messages"
).launch(share=True,inbrowser=True)"""


# tool to fetch the cryptocurrancy data from api


# coincap API endpoint
# Fetch cryptocurrency data
url = "https://api.coincap.io/v2/assets"
crypto_data = {}

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    assets = data.get("data", [])
    for asset in assets:
        crypto_data[asset['name'].lower()] = {
            "Symbol": asset['symbol'],
            "Price USD": round(float(asset['priceUsd']), 4)
        }
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

print(crypto_data)

def get_crypto_price(crypto_name):
    crypto_name = crypto_name.lower()
    if crypto_name in crypto_data:
        return crypto_data[crypto_name]
    else:
        return {"Error": "Cryptocurrency not found"}

price= get_crypto_price('bitcoin')

print(price)

# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_crypto_price",
    "description": "Retrieve the current price and symbol of a cryptocurrency. Use this whenever a user asks for the price of a specific cryptocurrency.",
    "parameters": {
        "type": "object",
        "properties": {
            "crypto_name": {
                "type": "string",
                "description": "The name of the cryptocurrency (e.g., 'bitcoin', 'ethereum'). Use lowercase names.",
            },
        },
        "required": ["crypto_name"],
        "additionalProperties": False
    }
}

tools = [
    {"type":"function","function":price_function}
]

# Function to handle tool calls
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    print(tool_call)
    arguments = json.loads(tool_call.function.arguments) # Use json.loads for string parsing
    print(arguments)
    # Get the price of the cryptocurrency
    crypto_name = arguments.get("crypto_name")
    price_data = get_crypto_price(crypto_name)  # Fetch price using your function
    price = price_data["Price USD"]
    symbol = price_data["Symbol"]
    response = {
        "role":"tool",
        "content": json.dumps({"name":crypto_name,"price":price}),
        "tool_call_id":tool_call.id
    }
    return response,price
# chat function
def chat(message,history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    
    response  = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    # check tools are called 
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, price = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content
        

gr.ChatInterface(fn=chat, type="messages").launch(share=True,inbrowser=True)
