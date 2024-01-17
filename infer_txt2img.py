import json
import logging
import os
import time
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

from infer_extra_single_image import sidebar_links

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
API_USERNAME = os.getenv("API_USERNAME")

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

    st.session_state.progress += 5
    progress_bar.progress(st.session_state.progress)

    while True:
        status_response = get_inference_job(inference["id"])
        st.json(status_response)
        status_response = status_response['data']
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
            break
        elif status_response['status'] == 'failed':
            st.error(f"Image generation failed.{status_response['sagemakerRaw']}")
            return
        else:
            time.sleep(2)

    for warning in st.session_state.warnings:
        st.warning(warning)

    return inference["id"]


def get_inference_job(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    url = API_URL + "inferences" + '/' + inference_id
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
        "user_id": API_USERNAME,
        "task_type": "txt2img",
        "inference_type": "Async",
        "models": {
            "Stable-diffusion": ["v1-5-pruned-emaonly.safetensors"],
            "embeddings": []
        },
        "filters": {
            "createAt": datetime.now().timestamp(),
            "creator": "sd-webui"
        }
    }

    st.info("payload for create inference job")
    st.json(body)

    job = requests.post(API_URL + "inferences", headers=headers, json=body)

    if job.status_code == 403:
        raise Exception(f"Your API URL or API KEY is not correct. Please check your .env file.")

    st.info(f"create inference job response\nPOST {API_URL}inferences")
    st.json(job.json())
    return job.json()


def run_inference_job(inference_id: str):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        'x-api-key': API_KEY
    }

    job = requests.put(API_URL + "inferences" + '/' + inference_id + '/start', headers=headers)

    return job.json()


def upload_inference_job_api_params(s3_url, positive: str):
    api_params = {
        "prompt": positive,
        "negative_prompt": "",
        "styles": [],
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0.0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 7.0,
        "width": 512,
        "height": 512,
        "restore_faces": None,
        "tiling": None,
        "do_not_save_samples": False,
        "do_not_save_grid": False,
        "eta": None,
        "denoising_strength": None,
        "s_min_uncond": 0.0,
        "s_churn": 0.0,
        "s_tmax": "Infinity",
        "s_tmin": 0.0,
        "s_noise": 1.0,
        "override_settings": {},
        "override_settings_restore_afterwards": True,
        "refiner_checkpoint": None,
        "refiner_switch_at": None,
        "disable_extra_networks": False,
        "comments": {},
        "enable_hr": False,
        "firstphase_width": 0,
        "firstphase_height": 0,
        "hr_scale": 2.0,
        "hr_upscaler": "Latent",
        "hr_second_pass_steps": 0,
        "hr_resize_x": 0,
        "hr_resize_y": 0,
        "hr_checkpoint_name": None,
        "hr_sampler_name": None,
        "hr_prompt": "",
        "hr_negative_prompt": "",
        "sampler_index": "DPM++ 2M Karras",
        "script_name": None,
        "script_args": [],
        "send_images": True,
        "save_images": False,
        "alwayson_scripts": {
            "refiner": {
                "args": [False, "", 0.8]
            },
            "seed": {
                "args": [-1, False, -1, 0, 0, 0]
            },
            "controlnet": {
                "args": [
                    {
                        "enabled": False,
                        "module": "none",
                        "model": "None",
                        "weight": 1,
                        "image": None,
                        "resize_mode": "Crop and Resize",
                        "low_vram": False,
                        "processor_res": -1,
                        "threshold_a": -1,
                        "threshold_b": -1,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "pixel_perfect": False,
                        "control_mode": "Balanced",
                        "is_ui": True,
                        "input_mode": "simple",
                        "batch_images": "",
                        "output_dir": "",
                        "loopback": False
                    },
                    {
                        "enabled": False,
                        "module": "none",
                        "model": "None",
                        "weight": 1,
                        "image": None,
                        "resize_mode": "Crop and Resize",
                        "low_vram": False,
                        "processor_res": -1,
                        "threshold_a": -1,
                        "threshold_b": -1,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "pixel_perfect": False,
                        "control_mode": "Balanced",
                        "is_ui": True,
                        "input_mode": "simple",
                        "batch_images": "",
                        "output_dir": "",
                        "loopback": False
                    },
                    {
                        "enabled": False,
                        "module": "none",
                        "model": "None",
                        "weight": 1,
                        "image": None,
                        "resize_mode": "Crop and Resize",
                        "low_vram": False,
                        "processor_res": -1,
                        "threshold_a": -1,
                        "threshold_b": -1,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "pixel_perfect": False,
                        "control_mode": "Balanced",
                        "is_ui": True,
                        "input_mode": "simple",
                        "batch_images": "",
                        "output_dir": "",
                        "loopback": False
                    }
                ]
            },
            "extra options": {
                "args": []
            }
        }
    }

    json_string = json.dumps(api_params)
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


if __name__ == "__main__":
    try:
        sidebar_links("txt2img")

        api_url = st.text_input("Please input API URL:", API_URL)
        api_key = st.text_input("Please input API KEY:", API_KEY)
        api_username = st.text_input("Please input API Username:", API_USERNAME)

        # User input
        prompt = st.text_input("Please input prompt:", "A cute dog")
        button = st.button('Generate Image')

        if button:
            API_URL = api_url
            API_KEY = api_key
            API_USERNAME = api_username
            st.session_state.warnings = []
            st.session_state.succeed_count = 0
            generate_lcm_image(prompt)
    except Exception as e:
        logger.exception(e)
        raise e
