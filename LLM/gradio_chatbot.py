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
    
    
system_prompt = " You are a helpful assistant chatbot"

def chat_openai(prompt):
    client = OpenAI()
    MODEL = 'gpt-4o-mini'
    msg = [
        {"role":"system", "content" : system_prompt},
        {"role": "user","content":prompt}
        
    ]
    completion = client.chat.completions.create(
        model= MODEL,
        messages=msg,
    )
    result = completion.choices[0].message.content
    
    return result


# here's a simple function

def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()

# gradio interface 
# Adding share=True means that it can be accessed publically
# A more permanent hosting is available using a platform called Spaces from HuggingFace, which we will touch on next week
# Adding inbrowser=True opens up a new browser window automatically
"""gr.Interface(fn=shout, 
            inputs="textbox", 
            outputs="textbox", 
            flagging_mode="never").launch(share=True,inbrowser=True)
"""
# Define this variable and then pass js=force_dark_mode when creating the Interface

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
"""gr.Interface(fn=shout, 
            inputs="textbox",
            outputs="textbox",
            flagging_mode="never", 
            js=force_dark_mode).launch(share=True,inbrowser=True)
"""

"""gradio_view = gr.Interface(
    fn=chat_openai,
    inputs=[gr.Textbox(label="Your message:",lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never",
    js=force_dark_mode).launch(share=True,inbrowser=True)

##################################################
# Let's use Markdown

gradio_view = gr.Interface(
    fn=chat_openai,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never",
    js=force_dark_mode).launch(share=True,inbrowser=True)"""

##################################################
# Let's create a call that streams back results
# If you'd like a refresher on Generators (the "yield" keyword),
# Please take a look at the Intermediate Python notebook in week1 folder.

def stream_gpt(prompt):
    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
      ]
    stream = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
        
view = gr.Interface(
    fn=stream_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never",
    js=force_dark_mode
)
view.launch(share=True,inbrowser=True)

##################################################