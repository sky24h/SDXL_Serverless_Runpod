import os

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import runpod
import base64
import signal
from inference_util import SDXL, check_data_format

model = SDXL()
timeout_s = 60 * 2


def encode_data(data_path):
    with open(data_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def handle_timeout(signum, frame):
    # raise an error when timeout, so that the serverless function will be terminated to avoid extra cost
    raise TimeoutError("Request Timeout! Please check the log for more details.")


def text2video(job):
    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(timeout_s)
        job_input = job["input"]
        job_input = check_data_format(job_input)
        print("prompt is '{}'".format(job_input["prompt"]))
        save_path = model.inference(
            prompt          = job_input["prompt"],
            steps           = job_input["steps"],
            width           = job_input["width"],
            height          = job_input["height"],
            n_prompt        = job_input["n_prompt"],
            guidance_scale  = job_input["guidance_scale"],
            high_noise_frac = job_input["high_noise_frac"],
            seed            = job_input["seed"],
            base_model      = job_input["base_model"],
            base_loras      = job_input["base_loras"],
        )
        image_data = encode_data(save_path)
        return {"filename": os.path.basename(save_path), "data": image_data}
    except Exception as e:
        return {"error": "Something went wrong, error message: {}".format(e)}
    finally:
        signal.alarm(0)


runpod.serverless.start({"handler": text2video})
