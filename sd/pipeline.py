import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5, 
             sampler_name="ddpm", n_inference_steps=50, models={}, seed=None,
             device=None, idle_device=None, tokenizer=None):
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 - 1")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)

        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)

        if seed is None:
            generate.seed()
        
        else:
            generator.manual_seed(seed)

        
        clip = models["clip"]
        clip = clip.to(device)

        if do_cfg:
            # convert prompt into tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            # (Batch, Sea_Len) --> Conversion into Tensor
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (Batch, Sea_Len) --> (Batch, Sea_Len, Dim)
            cond_context = clip(cond_tokens)


            # convert uncond_prompt into tokens
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids

            # (Batch, Sea_Len) --> Conversion into Tensor
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (Batch, Sea_Len) --> # (Batch, Sea_Len, Dim)
            uncond_context = clip(uncond_tokens)


            # combining both to a single prompt --> (2, Seq_Len, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])

        else:
            # convert prompt into tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (1, 77, 768)
            context = clip(tokens)

        to_idle(device)


        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)

        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)


        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))

            input_image_tensor = np.array(input_image_tensor)

            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # adds batch dim --> (Height, Width, Channel) --> (Batch, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)

            # (Batch, Height, Width, Channel) --> (Batch, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # adding noise
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run through encoder
            latents = encoder(input_image_tensor, encoder_noise)