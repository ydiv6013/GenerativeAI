import base64
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables
load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()
MODEL = "dall-e-3"

# Check the API key
if not api_key:
    print("No API key was found.")
else:
    print("API key found.")

# Define system prompt
system_prompt = (
    "Create a dynamic cityscape of {city}, highlighting its most iconic landmarks, tourist spots, "
    "and cultural uniqueness. The image should showcase the essence of the city with vibrant colors "
    "and in a lively pop-art style. Focus on blending modern and traditional elements of the city, "
    "with a visually appealing composition that captures the energy and atmosphere of the city. "
    "Think of a bold, colorful approach that emphasizes the city's character, its unique features, "
    "and the excitement of visiting this destination."
)

# Function to generate cityscape
def generate_cityscape(city):
    prompt = system_prompt.format(city=city)  # Format the city into the prompt
    try:
        image_response = openai.images.generate(
            model=MODEL,
            prompt=prompt,
            size="1024x1024",  # Valid size for DALL-E 3
            n=1,
            response_format="b64_json"
        )
        # Decode the image from base64
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        return f"An error occurred: {str(e)}"



# Create Gradio app
gr.Interface(
    fn=generate_cityscape,
    inputs=gr.Textbox(label="Enter a City Name"),
    outputs=gr.Image(label="Generated Cityscape"),
    title="Cityscape Generator",
    description="Enter the name of a city to generate a vibrant, pop-art style cityscape."
).launch(share=True)
