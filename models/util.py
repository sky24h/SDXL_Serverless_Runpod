import os
import torch


from safetensors import safe_open
from .convert_from_ckpt import convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint, convert_sdxl_text_encoder_2_checkpoint


def load_ckpt(model_path, device):
    if model_path.endswith(".ckpt"):
        state_dict = torch.load(model_path, map_location=device)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    elif model_path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(model_path, framework="pt", device=device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        raise ValueError(f"Unknown model type {model_path}, must be either .ckpt or .safetensors, otherwise convert it first")
    return state_dict


def load_base_model(pipeline, model_dir, base_model, device):
    # load base model
    base_model = os.path.join(model_dir, base_model)
    base_state_dict = load_ckpt(base_model, device)

    # convert base model to SDXL
    unet_sd, te1_sd, te2_sd, vae = {}, {}, {}, {}
    for k in list(base_state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k] = base_state_dict.pop(k)
        elif k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = base_state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = base_state_dict.pop(k)
        else:
            vae[k] = base_state_dict.pop(k)
    del base_state_dict

    infos = []
    print("loading from checkpoint")
    converted_unet = convert_ldm_unet_checkpoint(checkpoint=unet_sd, config=pipeline.unet.config)
    info = pipeline.unet.load_state_dict(converted_unet, strict=False)
    infos.append(info)
    # del converted_unet

    # text_model
    info = pipeline.text_encoder.load_state_dict(te1_sd, strict=False)
    infos.append(info)
    del te1_sd

    # text_model_2
    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
    info = pipeline.text_encoder_2.load_state_dict(converted_sd, strict=False)
    infos.append(info)
    del converted_sd

    # vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae, pipeline.vae.config)
    try:
        info = pipeline.vae.load_state_dict(converted_vae_checkpoint)
    except:
        converted_vae_checkpoint = {
            key.replace(".query.", ".to_q.").replace(".key.", ".to_k.").replace(".value.", ".to_v.").replace(".proj_attn.", ".to_out.0."): value
            for key, value in converted_vae_checkpoint.items()
        }
        info = pipeline.vae.load_state_dict(converted_vae_checkpoint)
    infos.append(info)
    del converted_vae_checkpoint

    print("Infos: {}".format(infos))

    return pipeline.to(device)


def apply_lora(pipeline, model_dir, lora_configs):
    for lora_name in lora_configs:
        lora_path, lora_scale = lora_configs[lora_name]
        lora_path = os.path.join(model_dir, lora_path)
        pipeline.load_lora_weights(lora_path, lora_scale=lora_scale)
        print("Lora applied, {} from {} with scale {}".format(lora_name, lora_path, lora_scale))
    return pipeline