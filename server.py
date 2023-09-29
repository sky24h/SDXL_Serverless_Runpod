import os

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import runpod
import base64
import signal
from inference_util import SDXL

model = SDXL()
timeout_s = 60


def encode_data(data_path):
    with open(data_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def handle_timeout(signum, frame):
    # raise an error when timeout, so that the serverless function will be terminated to avoid extra cost
    raise TimeoutError("Request Timeout! Please check the log for more details.")


def text2video(job):
    job_input = job["input"]
    prompt    = job_input["prompt"]
    steps     = job_input["steps"]
    height    = job_input["height"]
    width     = job_input["width"]
    print("prompt is '{}'".format(prompt))
    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(timeout_s)
        if not isinstance(prompt, str):
            return {"error": "The input is not valid."}
        else:
            save_path = model.inference(prompt=prompt, steps=steps, height=height, width=width)
            image_data = encode_data(save_path)
            return {"filename": os.path.basename(save_path), "data": image_data}
    except Exception as e:
        print(e)
        return {"error": "Something went wrong."}
    finally:
        signal.alarm(0)

runpod.serverless.start({"handler": text2video})
