import logging
import os

import requests
import streamlit as st
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# load .env file with specific name
load_dotenv(dotenv_path='.env')

# Your ApiGatewayUrl in Extension for Stable Diffusion
# Example: https://xxxx.execute-api.us-west-2.amazonaws.com/prod/
API_URL = os.getenv("API_URL")
# Your ApiGatewayUrlToken in Extension for Stable Diffusion
API_KEY = os.getenv("API_KEY")
# Your username in Extension for Stable Diffusion
# Some resources are limited to specific users
API_USERNAME = os.getenv("API_USERNAME", 'admin')

default_model = "v1-5-pruned-emaonly.safetensors"


def get_inference_job(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    url = API_URL + "inferences/" + inference_id
    job = requests.get(url, headers=headers)
    st.info(f"get status of inference job {url}")

    return job.json()


def generate_lcm_image(initial_prompt: str):
    st.spinner()
    st.session_state.progress = 5
    # Keep one progress bar instance for each column
    progress_bar = st.progress(st.session_state.progress)

    st.session_state.progress += 15
    progress_bar.progress(st.session_state.progress)

    generate_image(initial_prompt, progress_bar)
    st.session_state.succeed_count += 1
    progress_bar.empty()
    progress_bar.hidden = True


def run_inference_job(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    url = API_URL + "inferences" + '/' + inference_id + '/start'
    job = requests.put(url, headers=headers)
    st.info(f"payload for run inference job {url}")
    st.json(job.json())

    return job.json()


def sidebar_links(action: str):
    st.set_page_config(page_title=f"{action} - ESD", layout="wide")
    st.title(f"{action}")

    st.sidebar.image("https://d0.awsstatic.com/logos/powered-by-aws.png", width=200)
    st.sidebar.subheader("Extension for Stable Diffusion on AWS")
    st.sidebar.markdown(
        """
        - [extra-single-image](https://esd-extra-single-image.streamlit.app/)
        - [img2img](https://esd-img2img.streamlit.app/)
        - [lcm](https://esd-lcm.streamlit.app/)
        - [rembg](https://esd-rembg.streamlit.app/)
        - [txt2img](https://esd-txt2img.streamlit.app/)
        """
    )
