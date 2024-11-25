import requests
import json,os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr  


# define constants
load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
api_key = os.getenv('STOCK_API_KEY')
openai = OpenAI()
MODEL = "gpt-4o-mini"
    
# Check the API key
if not api_key:
    print("No API key was found.")
else:
    print("API key found.")
    

headers = {
	"x-rapidapi-key": api_key,
	"x-rapidapi-host": "indian-stock-exchange-api2.p.rapidapi.com"
	}

system_prompt = "You are a knowledgeable and helpful assistant specializing in the Indian stock exchange."
system_prompt += "Your primary role is to provide real-time stock prices, explain market trends, and assist users with stock market-related inquiries in a clear and concise manner."
system_prompt += "You can handle questions about specific stocks, market analysis, trading concepts, and general financial information related to the Indian stock market, including indices like the Nifty 50 and Sensex."
system_prompt += "Always ensure your responses are accurate, user-friendly, and engaging."
system_prompt += "If a user asks for real-time prices or data, remind them that external sources may be required to fetch the most up-to-date information."


print(system_prompt)

url = "https://indian-stock-exchange-api2.p.rapidapi.com/stock"

def get_stock_price(stock_name):
    querystring = {"name": str(stock_name)}
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        # Parse JSON response
        data = response.json()
        
        # Beautify JSON
        pretty_json = json.dumps(data, indent=4)
        print(f"Beautified JSON Response:\n{pretty_json}")
        
        # Convert JSON string to dictionary
        dictionary = json.loads(pretty_json)
        
        # Extract required information
        company_name = dictionary.get('companyName', 'N/A')
        current_price_nse = dictionary.get('currentPrice', {}).get('NSE', 'N/A')
        
        print(f"Company Name: {company_name}")
        print(f"Current Price (NSE): {current_price_nse}")
        
        # Return extracted information
        return {"companyName": company_name, "currentPriceNSE": current_price_nse}
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
stock_info = get_stock_price("Reliance Industries")
if stock_info:
    print(f"Stock Info: {stock_info}")
	

	