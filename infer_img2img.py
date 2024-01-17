import json
import logging
import os
import time
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

from lib import sidebar_links, Api

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


def generate_image(positive_prompts: str, progress_bar):
    job = create_inference_job()
    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)
    logger.info("job: {}".format(job))

    if job['statusCode'] == 400:
        st.error(job['message'])
        return

    inference = job['data']["inference"]

    upload_inference_job_api_params(inference["api_params_s3_upload_url"], positive_prompts)
    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    run_resp = api.run_inference_job(inference["id"])
    st.info("run inference job")
    st.json(run_resp)

    if 'errorMessage' in run_resp:
        st.error(run_resp['errorMessage'])
        return

    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    while True:
        status_response = api.get_inference_job(inference["id"])
        status_response = status_response['data']
        st.json(status_response)
        # if status is not created, increase the progress bar
        if status_response['status'] != 'created':
            if st.session_state.progress < 80:
                st.session_state.progress += 10
            progress_bar.progress(st.session_state.progress)
        logger.info("job status: {}".format(status_response['status']))
        if status_response['status'] == 'succeed':
            progress_bar.progress(100)
            st.info('data.img_presigned_urls')
            st.image(status_response['img_presigned_urls'][0], use_column_width=True)
            st.image(status_response['img_presigned_urls'][1], use_column_width=True)
            break
        elif status_response['status'] == 'failed':
            st.error(f"Image generation failed.{status_response['sagemakerRaw']}")
            return
        else:
            time.sleep(4)

    for warning in st.session_state.warnings:
        st.warning(warning)

    return inference["id"]


def create_inference_job():
    body = {
        'user_id': api.api_username,
        'task_type': 'img2img',
        'inference_type': 'Async',
        'models':
            {
                'Stable-diffusion': ['v1-5-pruned-emaonly.safetensors'],
                'VAE': ['Automatic'],
                'embeddings': []
            },
        'filters':
            {
                "createAt": datetime.now().timestamp(),
                'creator': 'sd-webui'
            }
    }

    return api.create_inference_job(body)


def upload_inference_job_api_params(s3_url, positive: str):
    with open('img2img_api_param.json') as f:
        api_params = json.load(f)
        f.close()

    api_params['prompt'] = positive
    json_string = json.dumps(api_params)
    response = requests.put(s3_url, data=json_string)
    response.raise_for_status()
    return response


if __name__ == "__main__":
    try:
        sidebar_links("img2img")

        # User input
        original_image = st.image("./cat.png")

        api_url = st.text_input("API URL:", API_URL)
        api_key = st.text_input("API KEY:", API_KEY)
        api_username = st.text_input("API Username:", API_USERNAME)

        prompt = st.text_input("What image do you want to update?", "dog face")
        button = st.button('Generate new Image')

        if button:
            api = Api(api_url, api_key, api_username)

            st.session_state.warnings = []
            st.session_state.succeed_count = 0
            generate_lcm_image(prompt)
    except Exception as e:
        logger.exception(e)
        st.error(e)
