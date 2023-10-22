import os
# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"


import tempfile
import PIL.Image
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import torch
import json
from omegaconf import OmegaConf
from models.util import load_base_model, apply_lora


def save_image(image: PIL.Image):
    output_video_path = tempfile.NamedTemporaryFile(suffix=".jpg").name
    image.save(output_video_path, "JPEG", quality=100, optimize=True, progressive=True)
    return output_video_path


def check_data_format(job_input):
    # must have prompt in the input, otherwise raise error to the user
    if "prompt" in job_input:
        prompt = job_input["prompt"]
    else:
        raise ValueError("The input must contain a prompt.")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")

    # optional params, make sure they are in the right format here, otherwise raise error to the user
    steps           = job_input["steps"] if "steps" in job_input else None
    width           = job_input["width"] if "width" in job_input else None
    height          = job_input["height"] if "height" in job_input else None
    n_prompt        = job_input["n_prompt"] if "n_prompt" in job_input else None
    guidance_scale  = job_input["guidance_scale"] if "guidance_scale" in job_input else None
    high_noise_frac = job_input["high_noise_frac"] if "high_noise_frac" in job_input else None
    seed            = job_input["seed"] if "seed" in job_input else None
    base_model      = job_input["base_model"] if "base_model" in job_input else None
    base_loras      = job_input["base_loras"] if "base_loras" in job_input else None

    # check optional params
    if steps is not None and not isinstance(steps, int):
        raise ValueError("steps must be an integer.")
    if width is not None and not isinstance(width, int):
        raise ValueError("width must be an integer.")
    if height is not None and not isinstance(height, int):
        raise ValueError("height must be an integer.")
    if n_prompt is not None and not isinstance(n_prompt, str):
        raise ValueError("n_prompt must be a string.")
    if guidance_scale is not None and not isinstance(guidance_scale, float) and not isinstance(guidance_scale, int):
        raise ValueError("guidance_scale must be a float or an integer.")
    if high_noise_frac is not None and not isinstance(high_noise_frac, float) and not isinstance(high_noise_frac, int):
        raise ValueError("high_noise_frac must be a float or an integer.")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if base_model is not None and not isinstance(base_model, str):
        raise ValueError("base_model must be a string.")
    if base_loras is not None:
        if not isinstance(base_loras, dict):
            raise ValueError("base_loras must be a dictionary.")
        for lora_name, lora_params in base_loras.items():
            if not isinstance(lora_name, str):
                raise ValueError("base_loras keys must be strings.")
            if not isinstance(lora_params, list):
                raise ValueError("base_loras values must be lists.")
            if len(lora_params) != 2:
                raise ValueError("base_loras values must be lists of length 2.")
            if not isinstance(lora_params[0], str):
                raise ValueError("base_loras values must be lists of strings.")
            if not isinstance(lora_params[1], float):
                raise ValueError("base_loras values must be lists of floats.")
    return {
        "prompt"         : prompt,
        "steps"          : steps,
        "width"          : width,
        "height"         : height,
        "n_prompt"       : n_prompt,
        "guidance_scale" : guidance_scale,
        "high_noise_frac": high_noise_frac,
        "seed"           : seed,
        "base_model"     : base_model,
        "base_loras"     : base_loras,
    }


class SDXL:
    def __init__(self):
        self.refiner_model_path    = os.path.join(os.path.dirname(__file__), "models/stable-diffusion-xl-refiner-1.0")
        self.pretrained_model_path = os.path.join(os.path.dirname(__file__), "models/stable-diffusion-xl-base-1.0")
        self.model_dir             = os.path.join(os.path.dirname(__file__), "models/custom-models")
        self.inference_config      = OmegaConf.load(os.path.join(os.path.dirname(__file__), "inference.yaml"))

        # can not be changed
        self.use_fp16 = True
        self.dtype    = torch.float16 if self.use_fp16 else torch.float32
        self.device   = "cuda"
        self.low_vram = (torch.cuda.mem_get_info()[1] / 1024 / 1024 / 1024) < 10
        print(f"GPU memory is lower than 12GB, use low_vram mode") if self.low_vram else None

        # load base model and refiner
        self._reload_base_model()
        self._reload_refiner()

        # pre-defined default params, can be changed
        self.steps           = 40
        self.high_noise_frac = 0.8
        self.guidance_scale  = 5.0
        self.person_prompts  = ["boy", "girl", "man", "woman", "person", "eye", "face"]

    def _reload_base_model(self, base_model="ORIGINAL_SDXL"):
        if hasattr(self, "base"):
            # release memory if low_vram
            self.base = self.base.to("cpu") if self.low_vram else self.base
        if base_model == "ORIGINAL_SDXL":
            # reload the original model
            self.base = StableDiffusionXLPipeline.from_pretrained(
                self.pretrained_model_path, torch_dtype = torch.float16, variant = "fp16", use_safetensors = True
            )
            print("Reloaded the original model.")
        elif base_model and base_model != "":
            # load custom model
            self.base = load_base_model(self.base, self.model_dir, base_model, "cpu")
            print(f"Loaded custom model: {base_model}")
        else:
            raise ValueError("base model must be specified")

    def _reload_refiner(self):
        if hasattr(self, "refiner"):
            # release memory if low_vram
            self.refiner = self.refiner.to("cpu") if self.low_vram else self.refiner
        self.refiner = DiffusionPipeline.from_pretrained(
            self.refiner_model_path,
            text_encoder_2  = self.base.text_encoder_2,
            vae             = self.base.vae,
            torch_dtype     = torch.float16,
            use_safetensors = True,
            variant         = "fp16",
        )

    def _get_model_params(self, prompt, width, height, n_prompt, base_model, base_loras):
        prompt = prompt[:-1] if prompt[-1] == "." else prompt
        if base_model is None:
            # when base_model is not specified, use the default model
            # if the prompt contains person-related keywords, use the person model, otherwise use the default model
            isPerson = False
            for keyword in self.person_prompts:
                if keyword in prompt:
                    isPerson = True
                    break

            # load default params
            model_config = self.inference_config.Person if isPerson else self.inference_config.Default
            base_model   = model_config.base_model
            base_loras   = model_config.base_loras
            prompt += ", "
            prompt += model_config.prompt
        else:
            # load default params
            model_config = self.inference_config.Default

        # update with user-specified params
        n_prompt = model_config.n_prompt if n_prompt is None else n_prompt
        width    = model_config.width if width is None else width
        height   = model_config.height if height is None else height

        return prompt, width, height, n_prompt, base_model, base_loras

    def _update_model(self, base_model, base_loras):
        # update model
        self._reload_base_model(base_model)

        # apply lora
        if base_loras:
            if len(base_loras) != 0:
                for lora in base_loras:
                    if len(base_loras[lora]) != 2:
                        raise ValueError('base_loras must be {"lora_name": [filename, scale], "lora_name2": [filename2, scale2] ...}')
                self.base = apply_lora(self.base, self.model_dir, base_loras)

    def inference(
        self,
        prompt,
        steps           = None,
        width           = None,
        height          = None,
        n_prompt        = None,
        guidance_scale  = None,
        high_noise_frac = None,
        seed            = None,
        base_model      = None,
        base_loras      = None,
    ):
        prompt, width, height, n_prompt, base_model, base_loras = self._get_model_params(
            prompt, width, height, n_prompt, base_model, base_loras
        )

        self._update_model(base_model, base_loras)

        torch.seed()
        torch.manual_seed(seed if seed is not None else torch.randint(0, 1000000000, (1,)).item())
        print(f"current seed: {torch.initial_seed()}")
        print(f"sampling {prompt} ...")
        print(f"negative prompt: {n_prompt}")
        # run both experts
        with torch.no_grad():
            self.base = self.base.to(self.device)
            image = self.base(
                prompt              = prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = self.steps if steps is None else steps,
                height              = height,
                width               = width,
                guidance_scale      = self.guidance_scale if guidance_scale is None else guidance_scale,
                denoising_end       = self.high_noise_frac if high_noise_frac is None else high_noise_frac,
                output_type         = "latent",
            ).images
            self.base = self.base.to("cpu") if self.low_vram else self.base # move base model to cpu to save gpu memory

            self.refiner = self.refiner.to(self.device)
            image = self.refiner(
                prompt              = prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = self.steps if steps is None else steps,
                guidance_scale      = self.guidance_scale if guidance_scale is None else guidance_scale,
                denoising_start     = self.high_noise_frac if high_noise_frac is None else high_noise_frac,
                image               = image,
                output_type         = "latent" if self.low_vram else "image",
            ).images

            if self.low_vram:
                # move refiner model to cpu to save gpu memory, except for vae which is needed for decoding
                self.refiner     = self.refiner.to("cpu")
                self.refiner.vae = self.refiner.vae.to(self.device)
                image            = self.refiner.vae.decode(image / self.refiner.vae.config.scaling_factor, return_dict=False)[0]
                self.refiner.vae = self.refiner.vae.to("cpu")
                image = self.refiner.image_processor.postprocess(image)[0]

            else:
                image = image[0]

        return save_image(image)


if __name__ == "__main__":
    # test code
    sdxl = SDXL()

    import json

    # simple config
    with open("test_input_simple.json", "r") as f:
        test_input = json.load(f)["input"]

    # only for testing
    test_input = check_data_format(test_input)

    # faster config
    save_path = sdxl.inference(prompt=test_input["prompt"])
    print("Result of simple config is saved to: {}\n".format(save_path))

    # complex custom config
    with open("./test_input.json", "r") as f:
        test_input = json.load(f)["input"]

    # only for testing
    test_input = check_data_format(test_input)

    # better config
    save_path = sdxl.inference(
        prompt          = test_input["prompt"],
        steps           = test_input["steps"],
        width           = test_input["width"],
        height          = test_input["height"],
        n_prompt        = test_input["n_prompt"],
        guidance_scale  = test_input["guidance_scale"],
        high_noise_frac = test_input["high_noise_frac"],
        seed            = test_input["seed"],
        base_model      = test_input["base_model"],
        base_loras      = test_input["base_loras"],
    )
    print("Result of custom config is saved to: {}\n".format(save_path))
