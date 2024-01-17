import base64
import json
import logging
import os
import time
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
# logging to stdout
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

    run_resp = run_inference_job(inference["id"])
    st.info("run inference job")
    st.json(run_resp)

    if 'errorMessage' in run_resp:
        st.error(run_resp['errorMessage'])
        return

    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    while True:
        status_response = get_inference_job(inference["id"])
        st.json(status_response)
        # if status is not created, increase the progress bar
        if status_response['data']['status'] != 'created':
            if st.session_state.progress < 80:
                st.session_state.progress += 10
            progress_bar.progress(st.session_state.progress)
        logger.info("job status: {}".format(status_response['data']['status']))
        if status_response['data']['status'] == 'succeed':
            progress_bar.progress(100)
            st.info("render data.img_presigned_urls")
            st.image(status_response['data']['img_presigned_urls'][0], use_column_width=True)
            break
        elif status_response['data']['status'] == 'failed':
            st.error(f"Image generation failed.{status_response['data']['sagemakerRaw']}")
            return
        else:
            time.sleep(4)

    for warning in st.session_state.warnings:
        st.warning(warning)

    return inference["id"]


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


def create_inference_job():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    body = {
        'user_id': 'admin',
        'task_type': 'extra-single-image',
        'inference_type': 'Async',
        'models':
            {
                'Stable-diffusion': ['v1-5-pruned-emaonly.safetensors'],
                # 'VAE': ['Automatic'],
                # 'embeddings': []
            },
        'filters':
            {
                "createAt": datetime.now().timestamp(),
                'creator': 'sd-webui'
            }
    }

    st.info("payload for create inference job")
    st.json(body)

    job = requests.post(API_URL + "inferences", headers=headers, json=body)
    st.info(f"create inference job response\nPOST {API_URL}inferences")
    st.json(job.json())

    if job.status_code == 403:
        raise Exception(f"Your API URL or API KEY is not correct. Please check your .env file.")

    return job.json()


def run_inference_job(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    job = requests.put(API_URL + 'inferences/' + inference_id + '/start', headers=headers)

    return job.json()


def upload_inference_job_api_params(s3_url, img_url: str):
    with open('extra-single-image-api-params.json') as f:
        api_params = json.load(f)
        f.close()

    st.info(f"get img form {img_url} as base64 string")
    response = requests.get(img_url)
    if response.status_code != 200:
        raise Exception(f"get img from {img_url} failed")

    img_base64 = base64.b64encode(response.content).decode('utf-8')
    api_params['image'] = img_base64

    json_string = json.dumps(api_params)

    st.info("api_params payload upload")
    st.json(api_params)

    response = requests.put(s3_url, data=json_string)
    response.raise_for_status()
    return response


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


def sidebar_links(action: str):
    st.set_page_config(page_title=f"{action} - ESD", layout="wide")
    st.title(f"{action} - ESD")
    st.sidebar.image("https://d0.awsstatic.com/logos/powered-by-aws.png", width=200)


if __name__ == "__main__":
    try:
        sidebar_links("extra-single-image")

        img_url = 'http://img.touxiangwu.com/2020/3/U3e6ny.jpg'
        original_image = st.image(img_url)

        api_url = st.text_input("API URL:", API_URL)
        api_key = st.text_input("API KEY:", API_KEY)
        api_username = st.text_input("API Username:", API_USERNAME)

        prompt = st.text_input("Please input image URL:", img_url)

        button = st.button('Generate new Image')

        if button:
            API_URL = api_url
            API_KEY = api_key
            API_USERNAME = api_username

            if not API_URL or not API_KEY or not API_USERNAME:
                raise Exception("API URL, API KEY and API Username can not be empty")

            st.session_state.warnings = []
            st.session_state.succeed_count = 0

            original_image.image(prompt)
            generate_lcm_image(prompt)
    except Exception as e:
        logger.exception(e)
        raise e
