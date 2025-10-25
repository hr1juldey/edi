import dspy
import asyncio
from ollama import AsyncClient # type: ignore

from PIL import Image
import base64
import io
import logging
from typing import List, Union


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# make the default ollama based connection setup
eye = dspy.LM(model="ollama/gemma3:4b")
brain = dspy.LM(model="ollama/qwen3:8b")

# make the fallback connction setup

def encode_images_base64(paths: Union[str, List[str]]) -> List[str]:
    """
    Encode one or more image files into base64 strings using Pillow.
    """
    if isinstance(paths, str):
        paths = [paths]

    encoded_images = []
    for path in paths:
        try:
            with Image.open(path) as img:
                buffer = io.BytesIO()
                img.save(buffer, format=img.format or "PNG")
                byte_data = buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8").replace("\n", "")
                encoded_images.append(base64_str)
        except Exception as e:
            print(f"[WARN] Skipping '{path}' - {e}")

    return encoded_images


async def chat(RQ: str, IMP: Union[str, List[str]], model: str = "gemma3:4b"):
    """
    Sends a text prompt (RQ) and one or more image paths (IMP) to a vision model.
    Automatically base64-encodes images before sending.
    """
    # Encode image(s) to base64
    image_b64_list = encode_images_base64(IMP)

    message = {
        "role": "user",
        "content": RQ,
        "images": image_b64_list
    }

    response = await AsyncClient().chat(model=model, messages=[message])
    description =response['message']['content']
    return description


def setup_llm_models():
    """
    Setup and return the eye (VLM) and brain (reasoning LLM) models.
    
    Returns:
        tuple: (eye_model, brain_model) as dspy.LM instances
    """
    logger.debug("Setting up LLM models")
    eye_model = dspy.LM(model="ollama/gemma3:4b")
    brain_model = dspy.LM(model="ollama/qwen3:8b")
    logger.debug("LLM models setup completed")
    return eye_model, brain_model


def configure_dspy(brain_model):
    """
    Configure DSPy with the brain model.
    
    Args:
        brain_model (dspy.LM): The brain model to configure DSPy with
    """
    logger.debug(f"Configuring DSPy with brain model: {brain_model}")
    dspy.configure(lm=brain_model)
    logger.debug("DSPy configuration completed")


if __name__ == "__main__":
    OP = asyncio.run(chat(
        RQ="Systematically segment and define all entities and components with position, color (hex), and detailed description. ONLY AND ONLY return valid structured JSON",
        IMP="/home/riju279/Documents/Code/Zonko/Interpreter/interpreter/IP.jpeg",
        model="gemma3:4b"
    ))
    print(OP)