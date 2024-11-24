import os
from dotenv import load_dotenv
from IPython.display import Markdown,display,update_display
from openai import OpenAI
import gradio as gr

load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
api_key = os.getenv('OPENAI_API_KEY')

print(api_key)

# check the key
if not api_key : 
    print("No API key was found ")
else : 
    print("API key found ")
    
    
    
openai = OpenAI()
model = 'gpt-4o-mini'

system_prompt = "You are a Helpful Ecommerce assistant"

# open ai completions data format
"""
[
    {"role": "system", "content": "system message here"},
    {"role": "user", "content": "first user prompt here"},
    {"role": "assistant", "content": "the assistant's response"},
    {"role": "user", "content": "the new user prompt"},
]
    
"""

"""# chat function

def chat(message,history):
    message = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    
    stream  = openai.chat.completions.create(
        model=model,
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


####################################################


# create a responve based on certain word in the user_promot
# let say user asking for invoice

def chat_invoice(message, history):
    # Combine system prompt, history, and new message
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    
    # Check if the message contains 'invoice'
    if 'invoice' in message.lower():
        # Add a helpful response to the assistant
        messages.append({"role": "assistant", "content": "You should ask for the invoice number, name, and purchased products."})
    
    # Stream the OpenAI response
    stream = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ''
        yield result
    

# use gradio chatinterface for ui
gr.ChatInterface(
    fn=chat_invoice,
    type="messages"
).launch(share=True,inbrowser=True)