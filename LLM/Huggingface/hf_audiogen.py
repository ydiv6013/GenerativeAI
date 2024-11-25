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


# text to speech pipeline

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts",device=device)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
