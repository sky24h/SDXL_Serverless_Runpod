import os
import time
import numpy as np
import tempfile
import PIL.Image
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import torch
import json
from safetensors import safe_open


def save_image(image: PIL.Image):
    output_video_path = tempfile.NamedTemporaryFile(suffix=".jpg").name
    image.save(output_video_path, "JPEG", quality=100, optimize=True, progressive=True)
    return output_video_path


class SDXL:
    def __init__(self):
        base_model_path = os.path.join(os.path.dirname(__file__), "models/stable-diffusion-xl-base-1.0")
        refiner_model_path = os.path.join(os.path.dirname(__file__), "models/stable-diffusion-xl-refiner-1.0")
        self.base = StableDiffusionXLPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(
            "cuda"
        )
        self.refiner = DiffusionPipeline.from_pretrained(
            refiner_model_path,
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        # Define what % of steps to be run on each experts (80/20) here
        self.high_noise_frac = 0.8

        self.person_prompts = ["boy", "girl", "man", "woman", "person", "eye", "face"]
        with open("preset_prompts.json", "r") as f:
            self.preset_prompts = json.load(f)

    def _update_model(self, model_type):
        if model_type == "Person":
            # !TODO, update to specific model for person here
            self.base.load_lora_weights("./nudify_xl.safetensors")
            pass
        elif model_type == "Scene":
            # !TODO, update to specific model for scene here
            pass

    def check_prompt(self, prompt):
        isPerson = False
        for keyword in self.person_prompts:
            if keyword in prompt:
                isPerson = True
                break
        if prompt.endswith("."):
            prompt = prompt[:-1] + ","
        else:
            prompt += ", "

        # update model
        if isPerson:
            prompt += self.preset_prompts["Person"]["prompt"]
            n_prompt = self.preset_prompts["Person"]["n_prompt"]
            self._update_model("Person")
        else:
            prompt += self.preset_prompts["Scene"]["prompt"]
            n_prompt = self.preset_prompts["Scene"]["n_prompt"]
            self._update_model("Scene")
        return prompt, n_prompt, isPerson

    def inference(self, prompt, steps, height=1024, width=1024, seed=None):
        prompt, n_prompt, isPerson = self.check_prompt(prompt)
        if not isPerson:
            width, height = height, width

        torch.seed()
        torch.manual_seed(seed if seed is not None else torch.randint(0, 1000000000, (1,)).item())

        # run both experts
        image = self.base(
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images

        image = self.refiner(
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]

        return save_image(image)


if __name__ == "__main__":
    # test code
    model = SDXL()

    fixed_seed = 253087639#torch.randint(0, 1000000000, (1,)).item()
    print("current seed: {}".format(fixed_seed))

    import json
    with open("test_input.json", "r") as f:
        test_input = json.load(f)["input"]
    print("Prompt: {}".format(test_input["prompt"]))

    time_start = time.time()
    res_1 = model.inference(prompt=test_input["prompt"], steps=test_input["steps"], width=test_input["width"], height=test_input["height"], seed=fixed_seed)
    print("Inference time: {}".format(time.time() - time_start))
    print("Result saved to {}".format(res_1))
