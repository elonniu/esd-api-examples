import logging

import requests
import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

default_model = "v1-5-pruned-emaonly.safetensors"


class Api:

    def __init__(self, api_url: str, api_key: str, api_username: str, inference_type: str):

        if not api_url or not api_key or not api_username:
            raise Exception("API URL, API KEY and API Username can not be empty")

        self.api_url = api_url
        self.api_key = api_key
        self.api_username = api_username
        self.inference_type = inference_type

    def get_inference_job(self, inference_id: str):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            'x-api-key': self.api_key
        }

        url = self.api_url + "inferences/" + inference_id
        job = requests.get(url, headers=headers)
        st.info(f"get status of inference job GET {url}")
        st.json(job.json(), expanded=False)

        return job.json()

    def start_inference_job(self, inference_id: str):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            'x-api-key': self.api_key
        }

        url = self.api_url + 'inferences/' + inference_id + '/start'
        job = requests.put(url, headers=headers)
        st.info(f"start inference job response PUT {url}")
        st.json(job.json(), expanded=False)

        return job.json()

    def create_inference_job(self, body):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            'x-api-key': self.api_key
        }

        st.info("payload for create inference job")
        st.json(body, expanded=False)

        job = requests.post(self.api_url + "inferences", headers=headers, json=body)
        st.info(f"create inference job response\nPOST {self.api_url}inferences")
        st.json(job.json(), expanded=False)

        if job.status_code == 403:
            raise Exception(f"Your API URL or API KEY is not correct. Please check your .env file.")

        return job.json()


def sidebar_links(action: str):
    st.set_page_config(page_title=f"{action} - ESD", layout="wide")
    st.title(f"{action}")

    st.sidebar.image("https://d0.awsstatic.com/logos/powered-by-aws.png", width=200)
    st.sidebar.subheader("Extension for Stable Diffusion on AWS")
    st.sidebar.markdown(
        """
        - [txt2img](https://esd-txt2img.streamlit.app/)
        - [txt2img lcm](https://esd-lcm.streamlit.app/)
        - [img2img](https://esd-img2img.streamlit.app/)
        - [extra-single-image](https://esd-extra-single-image.streamlit.app/)
        - [rembg](https://esd-rembg.streamlit.app/)
        """
    )
