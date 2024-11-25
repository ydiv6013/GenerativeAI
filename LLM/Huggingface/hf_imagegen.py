import torch,os
from huggingface_hub import login
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio
from dotenv import load_dotenv
from PIL import Image

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# define constants
load_dotenv(dotenv_path='/Users/yogesh/pythoncode/GenerativeAI/LLM/.env')
hf_token = os.getenv('HF_API_KEY')

# Check the API key
if not hf_token:
    print("No API key was found.")
else:
    print("API key found.")

#login to hugging face 
login(hf_token,add_to_git_credential=True)


# Check for device compatibility (Apple MPS or CPU fallback)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the Stable Diffusion pipeline
image_gen = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    use_safetensors = True,
    variant = "fp16"
).to(device)

# Generate an image
prompt = "An astronaut riding a green horse"
print("Generating image...")
result = image_gen(prompt=prompt,width=256,height=256)


# save and preview the generated image
image = result.images[0]
output_path = "/Users/yogesh/pythoncode/GenerativeAI/LLM/Huggingface/GeneratedImage.png"
image.save(output_path)
print(f"Image saved to {output_path}")

# preview the image
image.show()