import base64
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

    run_resp = api.start_inference_job(inference["id"])

    if api.inference_type == 'Real-time':
        st.info("render data.img_presigned_urls")
        st.image(run_resp['data']['img_presigned_urls'][0], use_column_width=True)
        return

    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    while True:
        status_response = api.get_inference_job(inference["id"])
        status_response = status_response['data']
        # if status is not created, increase the progress bar
        if status_response['status'] != 'created':
            if st.session_state.progress < 80:
                st.session_state.progress += 10
            progress_bar.progress(st.session_state.progress)
        logger.info("job status: {}".format(status_response['status']))
        if status_response['status'] == 'succeed':
            progress_bar.progress(100)
            st.info("render data.img_presigned_urls")
            st.image(status_response['img_presigned_urls'][0], use_column_width=True)
            break
        elif status_response['status'] == 'failed':
            st.error(f"Image generation failed.{status_response['sagemakerRaw']}")
            return
        else:
            time.sleep(4)

    for warning in st.session_state.warnings:
        st.warning(warning)


def create_inference_job():
    body = {
        'user_id': api.api_username,
        'task_type': 'rembg',
        'inference_type': api.inference_type,
        'models':
            {
                'Stable-diffusion': ['v1-5-pruned-emaonly.safetensors'],
            },
        'filters':
            {
                "createAt": datetime.now().timestamp(),
                'creator': 'sd-webui'
            }
    }

    return api.create_inference_job(body)


def upload_inference_job_api_params(s3_url, img_url: str):
    with open('rembg-api-params.json') as f:
        api_params = json.load(f)
        f.close()

    st.info(f"get img form {img_url} as base64 string")
    response = requests.get(img_url)
    if response.status_code != 200:
        raise Exception(f"get img from {img_url} failed")

    img_base64 = base64.b64encode(response.content).decode('utf-8')
    api_params['input_image'] = img_base64

    json_string = json.dumps(api_params)

    st.info("api_params payload upload")
    st.json(api_params, expanded=False)

    response = requests.put(s3_url, data=json_string)
    response.raise_for_status()
    return response


if __name__ == "__main__":
    try:
        sidebar_links("rembg")

        # User input
        img_url = 'https://img2.baidu.com/it/u=854943903,3669169186&fm=253&fmt=auto&app=138&f=JPEG?w=750&h=500'
        original_image = st.image(img_url)

        api_url = st.text_input("API URL:", API_URL)
        api_key = st.text_input("API KEY:", API_KEY)
        api_username = st.text_input("API Username:", API_USERNAME)

        prompt = st.text_input("Please input image URL:", img_url)

        inference_type = st.radio("Inference Type", ('Async', 'Real-time'), horizontal=True)

        button = st.button('Generate new Image')

        if button:
            api = Api(api_url, api_key, api_username, inference_type)
            original_image.image(prompt)
            st.session_state.warnings = []
            st.session_state.succeed_count = 0
            generate_lcm_image(prompt)
    except Exception as e:
        logger.exception(e)
        st.error(e)
