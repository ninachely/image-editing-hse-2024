"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os

import torch
import torch.nn.functional as F
import tqdm
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from torch import nn
from transformers import CLIPTokenizer
from transformers.activations import QuickGELUActivation as QuickGELU

from lavis.common.registry import registry
from lavis.common.utils import download_and_untar, is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_diffusion_models.modeling_ctx_clip import CtxCLIPTextModel
from lavis.models.blip_diffusion_models.utils import numpy_to_pil, prepare_cond_image
from lavis.models.blip_diffusion_models.ptp_utils import (
    LocalBlend,
    P2PCrossAttnProcessor,
    AttentionRefine,
    AttentionStore
)

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
from lavis.models.blip_diffusion_models.ptp_utils_orig import (
    text_under_image,
    view_images,
    diffusion_step,
    latent2image,
    init_latent,
    text2image_ldm,
    text2image_ldm_stable,
    register_attention_control,
    get_word_inds,
    update_alpha_time_word,
    get_time_words_attention_alpha,
    register_attention_control,
    slerp_tensor
)
import lavis.models.blip_diffusion_models.seq_aligner
import accelerate
from torch.optim.adam import Adam
from PIL import Image

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x
    

@registry.register_model("blip_diffusion")
class BlipDiffusion(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip-diffusion/blip_diffusion_base.yaml",
        "canny": "configs/models/blip-diffusion/blip_diffusion_controlnet_canny.yaml",
        "depth": "configs/models/blip-diffusion/blip_diffusion_controlnet_depth.yaml",
        "hed": "configs/models/blip-diffusion/blip_diffusion_controlnet_hed.yaml",
    }

    def __init__(
        self,
        vit_model="clip_L",
        qformer_num_query_token=16,
        qformer_cross_attention_freq=1,
        qformer_pretrained_path=None,
        qformer_train=False,
        sd_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        sd_train_text_encoder=False,
        controlnet_pretrained_model_name_or_path=None,
        vae_half_precision=False,
        proj_train=False,
    ):
        super().__init__()

        self.num_query_token = qformer_num_query_token

        # BLIP-2
        self.blip = Blip2Qformer(
            vit_model=vit_model,
            num_query_token=qformer_num_query_token,
            cross_attention_freq=qformer_cross_attention_freq,
        )
        if qformer_pretrained_path is not None:
            state_dict = torch.load(qformer_pretrained_path, map_location="cpu")[
                "model"
            ]
            # qformer keys: Qformer.bert.encoder.layer.1.attention.self.key.weight
            # ckpt keys: text_model.bert.encoder.layer.1.attention.self.key.weight
            for k in list(state_dict.keys()):
                if "text_model" in k:
                    state_dict[k.replace("text_model", "Qformer")] = state_dict.pop(k)

            msg = self.blip.load_state_dict(state_dict, strict=False)
            assert all(["visual" in k for k in msg.missing_keys])
            assert len(msg.unexpected_keys) == 0

        self.qformer_train = qformer_train

        # projection layer
        self.proj_layer = ProjLayer(
            in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
        )
        self.proj_train = proj_train

        # stable diffusion
        self.tokenizer = CLIPTokenizer.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CtxCLIPTextModel.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="vae"
        )
        if vae_half_precision:
            self.vae.half()

        self.unet = UNet2DConditionModel.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="unet"
        )
        # self.unet.enable_xformers_memory_efficient_attention()

        self.noise_scheduler = DDPMScheduler.from_config(
            sd_pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.sd_train_text_encoder = sd_train_text_encoder

        if controlnet_pretrained_model_name_or_path is not None:
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_pretrained_model_name_or_path
            )

        self.freeze_modules()

        self.ctx_embeddings_cache = nn.Parameter(
            torch.zeros(1, self.num_query_token, 768), requires_grad=False
        )
        self._use_embeddings_cache = False

        # inference-related
        self._CTX_BEGIN_POS = 2

    def freeze_modules(self):
        to_freeze = [self.vae]
        if not self.sd_train_text_encoder:
            to_freeze.append(self.text_encoder)

        if not self.qformer_train:
            to_freeze.append(self.blip)

        if not self.proj_train:
            to_freeze.append(self.proj_layer)

        for module in to_freeze:
            module.eval()
            module.train = self.disabled_train
            module.requires_grad_(False)

    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    @property
    def pndm_scheduler(self):
        if not hasattr(self, "_pndm_scheduler"):
            self._pndm_scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                set_alpha_to_one=False,
                skip_prk_steps=True,
            )
        return self._pndm_scheduler

    @property
    def ddim_scheduler(self):
        if not hasattr(self, "_ddim_scheduler"):
            self._ddim_scheduler = DDIMScheduler.from_config(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )
        return self._ddim_scheduler

    def before_training(self, dataset, **kwargs):
        assert len(dataset) == 1, "Only support single dataset for now."

        key = list(dataset.keys())[0]
        dataset = dataset[key]["train"]

        # collect all examples
        # [FIXME] this is not memory efficient. may OOM if the dataset is large.
        examples = [dataset[i] for i in range(dataset.len_without_repeat)]
        input_images = (
            torch.stack([example["inp_image"] for example in examples])
            .to(memory_format=torch.contiguous_format)
            .float()
        ).to(self.device)
        subject_text = [dataset.subject for _ in range(input_images.shape[0])]

        # calculate ctx embeddings and cache them
        ctx_embeddings = self.forward_ctx_embeddings(
            input_image=input_images, text_input=subject_text
        )
        # take mean of all ctx embeddings
        ctx_embeddings = ctx_embeddings.mean(dim=0, keepdim=True)
        self.ctx_embeddings_cache = nn.Parameter(ctx_embeddings, requires_grad=True)
        self._use_embeddings_cache = True

        # free up CUDA memory
        self.blip.to("cpu")
        self.proj_layer.to("cpu")

        torch.cuda.empty_cache()

    def forward(self, samples):
        latents = self.vae.encode(samples["tgt_image"].half()).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        ctx_embeddings = self.forward_ctx_embeddings(
            input_image=samples["inp_image"], text_input=samples["subject_text"]
        )

        # Get the text embedding for conditioning
        input_ids = self.tokenizer(
            samples["caption"],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=[self._CTX_BEGIN_POS] * input_ids.shape[0],
        )[0]

        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents.float(), timesteps, encoder_hidden_states
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return {"loss": loss}

    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

    def _build_prompts_edit(self, cond_subject, tgt_subject, prompt):
        placeholder = " ".join(["sks"] * self.num_query_token)

        src_prompt = f"a {cond_subject} {prompt}"
        tgt_prompt = f"a {placeholder} {tgt_subject} {prompt}"

        return [src_prompt, tgt_prompt]

    def _predict_noise(
        self,
        t,
        latent_model_input,
        text_embeddings,
        width=512,
        height=512,
        cond_image=None,
    ):
        if hasattr(self, "controlnet"):
            cond_image = prepare_cond_image(
                cond_image, width, height, batch_size=1, device=self.device
            )

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond_image,
                # conditioning_scale=controlnet_condition_scale,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = None, None

        noise_pred = self.unet(
            latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )["sample"]

        return noise_pred

    def _init_latent(self, latent, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=generator.device,
            )
        latent = latent.expand(
            batch_size,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )
        return latent.to(self.device)

    def _forward_prompt_embeddings(self, input_image, src_subject, prompt):
        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.forward_ctx_embeddings(input_image, src_subject)

        # 2. embeddings for prompt, with query_embeds as context
        tokenized_prompt = self._tokenize_text(prompt).to(self.device)
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self._CTX_BEGIN_POS],
        )[0]

        return text_embeddings

    @torch.no_grad()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        assert isinstance(image, torch.Tensor)

        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    def _inversion_transform(self, image, target_size=512):
        from torchvision import transforms

        tform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
            ]
        )
        image = tform(image).unsqueeze(0).to(self.device)
        return 2.0 * image - 1.0

    # @torch.no_grad()
    def edit(
        self,
        samples,
        lb_threshold=0.3,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=50,
        num_inversion_steps=50,
        neg_prompt="",
    ):
        # torch.Tensor of shape torch.Size([1, 3, 512, 512]) -> PIL.Image.Image
        raw_image = samples["raw_image"] # PIL.Image.Image

        prompt = f'a {samples["src_subject"]} {samples["prompt"][0].strip()}'

        #  scheduler=self.ddim_scheduler,
        null_inversion = NegativePromptInversion(device=self.device,
                                                 scheduler=self.ddim_scheduler,
                                                 tokenizer=self.tokenizer, 
                                                 text_encoder=self.text_encoder, 
                                                 vae=self.vae, 
                                                 unet=self.unet, 
                                                 noise_scheduler=self.noise_scheduler, 
                                                 num_ddim_steps=num_inversion_steps,
                                                 guidance_scale=guidance_scale)

        # (image_gt, image_enc), x_t, uncond_embeddings
        (image_gt, image_rec, image_rec_latent), ddim_latents, uncond_embeddings = null_inversion.invert(raw_image, prompt, verbose=True)

        # inv_latents -> x_t
        recon_image = self.generate_then_edit(
            samples=samples,
            latents=ddim_latents,
            seed=seed,
            neg_prompt=neg_prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            use_inversion=True,
            lb_threshold=lb_threshold,
        )

        return recon_image

    @torch.no_grad()
    def _ddim_inverse(
        self,
        samples,
        latents,
        guidance_scale=1.0,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=50,
    ):
        src_subject = samples["src_subject"]  # source subject category
        prompt = samples["prompt"]

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=src_subject,
            prompt_strength=1.0,
            prompt_reps=1,
        )

        tokenized_prompt = self._tokenize_text(prompt, with_query=False).to(self.device)
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=None,
        )[0]

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = self._init_latent(latents, height, width, generator, batch_size=1)

        scheduler = self.ddim_scheduler

        # set timesteps
        extra_set_kwargs = {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        iterator = tqdm.tqdm(reversed(scheduler.timesteps))

        for i, t in enumerate(iterator):
            latents = self._noise_latent_step(
                latents=latents,
                t=t,
                text_embeddings=text_embeddings,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )

        return latents

    @torch.no_grad()
    def generate(
        self,
        samples,
        latents=None,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=50,
        neg_prompt="",
        controller=None,
        prompt_strength=1.0,
        prompt_reps=20,
        use_ddim=False,
    ):
        if controller is not None:
            self._register_attention_refine(controller)

        cond_image = samples["cond_images"]  # reference image
        cond_subject = samples["cond_subject"]  # source subject category
        tgt_subject = samples["tgt_subject"]  # target subject category
        prompt = samples["prompt"]
        cldm_cond_image = samples.get("cldm_cond_image", None)  # conditional image

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=tgt_subject,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )

        text_embeddings = self._forward_prompt_embeddings(
            cond_image, cond_subject, prompt
        )

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None,
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = self._init_latent(latents, height, width, generator, batch_size=1)

        scheduler = self.pndm_scheduler if not use_ddim else self.ddim_scheduler

        # set timesteps
        extra_set_kwargs = {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        iterator = tqdm.tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            latents = self._denoise_latent_step(
                latents=latents,
                t=t,
                text_embeddings=text_embeddings,
                cond_image=cldm_cond_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                use_inversion=use_ddim,
            )

        image = self._latent_to_image(latents)

        return image

    def _register_attention_refine(
        self,
        src_subject,
        prompts,
        num_inference_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        threshold=0.3,
    ):
        device, tokenizer = self.device, self.tokenizer

        lb = LocalBlend(
            prompts=prompts,
            words=(src_subject,),
            device=device,
            tokenizer=tokenizer,
            threshold=threshold,
        )

        controller = AttentionRefine(
            prompts,
            num_inference_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            tokenizer=tokenizer,
            device=device,
            local_blend=lb,
        )

        self._register_attention_control(controller)

        return controller

    def _register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        if controller is not None:
            controller.num_att_layers = cross_att_count

    @torch.no_grad()
    def generate_then_edit(
        self,
        samples,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        guidance_scale=7.5,
        height=512,
        width=512,
        latents=None,
        seed=42,
        num_inference_steps=250,
        neg_prompt="",
        use_inversion=False,
        lb_threshold=0.3,
    ):
        cond_image = samples["cond_images"]  # reference image
        cond_subject = samples["cond_subject"]  # source subject category

        src_subject = samples["src_subject"]
        tgt_subject = samples["tgt_subject"]  # target subject category

        prompt = samples["prompt"]
        assert len(prompt) == 1, "Do not support multiple prompts for now"
        prompt = self._build_prompts_edit(src_subject, tgt_subject, prompt[0])
        print(prompt)

        controller = self._register_attention_refine(
            src_subject=src_subject,
            prompts=prompt,
            num_inference_steps=num_inference_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            threshold=lb_threshold,
        )

        query_embeds = self.forward_ctx_embeddings(cond_image, cond_subject)

        tokenized_prompt_bef = self._tokenize_text(prompt[:1], with_query=False).to(
            self.device
        )
        tokenized_prompt_aft = self._tokenize_text(prompt[1:], with_query=True).to(
            self.device
        )

        text_embeddings_bef = self.text_encoder(
            input_ids=tokenized_prompt_bef.input_ids,
        )[0]
        text_embeddings_aft = self.text_encoder(
            input_ids=tokenized_prompt_aft.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self._CTX_BEGIN_POS],
        )[0]

        text_embeddings = torch.cat([text_embeddings_bef, text_embeddings_aft], dim=0)

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0

        # [TODO] add support for batched input
        batch_size = 2

        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # FIXME use context embedding for uncond_input or not?
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None,
            )[0]
            # repeat the uncond embedding to match the number of prompts
            uncond_embeddings = uncond_embeddings.expand(batch_size, -1, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = self._init_latent(latents, height, width, generator, batch_size)

        scheduler = self.pndm_scheduler if not use_inversion else self.ddim_scheduler
        # set timesteps
        scheduler.set_timesteps(num_inference_steps)

        iterator = tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            latents = self._denoise_latent_step(
                latents=latents,
                t=t,
                text_embeddings=text_embeddings,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                use_inversion=use_inversion,
            )

            latents = controller.step_callback(latents)

        image = self._latent_to_image(latents)
        controller.reset()

        return image

    def _latent_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = numpy_to_pil(image)

        return image

    def _noise_latent_step(
        self,
        latents,
        t,
        text_embeddings,
        guidance_scale,
        height,
        width,
    ):
        def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
            """from noise to image"""
            return (
                alpha_tm1**0.5
                * (
                    (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
                    + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
                )
                + x_t
            )

        do_classifier_free_guidance = guidance_scale > 1.0

        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        # predict the noise residual
        noise_pred = self._predict_noise(
            t=t,
            latent_model_input=latent_model_input,
            text_embeddings=text_embeddings,
            width=width,
            height=height,
        )

        scheduler = self.ddim_scheduler

        prev_timestep = (
            t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else scheduler.final_alpha_cumprod
        )
        alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
        latents = backward_ddim(
            x_t=latents,
            alpha_t=alpha_prod_t,
            alpha_tm1=alpha_prod_t_prev,
            eps_xt=noise_pred,
        )

        return latents

    def _denoise_latent_step(
        self,
        latents,
        t,
        text_embeddings,
        guidance_scale,
        height,
        width,
        cond_image=None,
        use_inversion=False,
    ):
        if use_inversion:
            noise_placeholder = []

        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        # predict the noise residual
        noise_pred = self._predict_noise(
            t=t,
            latent_model_input=latent_model_input,
            text_embeddings=text_embeddings,
            width=width,
            height=height,
            cond_image=cond_image,
        )

        if use_inversion:
            noise_placeholder.append(noise_pred[2].unsqueeze(0))

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if use_inversion:
            noise_placeholder.append(noise_pred[-1].unsqueeze(0))
            noise_pred = torch.cat(noise_placeholder)

        # compute the previous noisy sample x_t -> x_t-1
        scheduler = self.ddim_scheduler if use_inversion else self.pndm_scheduler

        latents = scheduler.step(
            noise_pred,
            t,
            latents,
        )["prev_sample"]

        return latents

    def _tokenize_text(self, text_input, with_query=True):
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        if with_query:
            max_len -= self.num_query_token

        tokenized_text = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return tokenized_text

    def forward_ctx_embeddings(self, input_image, text_input, ratio=None):
        def compute_ctx_embeddings(input_image, text_input):
            # blip_embeddings = self.blip(image=input_image, text=text_input)
            blip_embeddings = self.blip.extract_features(
                {"image": input_image, "text_input": text_input}, mode="multimodal"
            ).multimodal_embeds
            ctx_embeddings = self.proj_layer(blip_embeddings)

            return ctx_embeddings

        if isinstance(text_input, str):
            text_input = [text_input]

        if self._use_embeddings_cache:
            # expand to batch size
            ctx_embeddings = self.ctx_embeddings_cache.expand(len(text_input), -1, -1)
        else:
            if isinstance(text_input[0], str):
                text_input, input_image = [text_input], [input_image]

            all_ctx_embeddings = []

            for inp_image, inp_text in zip(input_image, text_input):
                ctx_embeddings = compute_ctx_embeddings(inp_image, inp_text)
                all_ctx_embeddings.append(ctx_embeddings)

            if ratio is not None:
                assert len(ratio) == len(all_ctx_embeddings)
                assert sum(ratio) == 1
            else:
                ratio = [1 / len(all_ctx_embeddings)] * len(all_ctx_embeddings)

            ctx_embeddings = torch.zeros_like(all_ctx_embeddings[0])

            for ratio, ctx_embeddings_ in zip(ratio, all_ctx_embeddings):
                ctx_embeddings += ratio * ctx_embeddings_

        return ctx_embeddings

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "clip_L")

        qformer_cross_attention_freq = cfg.get("qformer_cross_attention_freq", 1)
        qformer_num_query_token = cfg.get("qformer_num_query_token", 16)
        qformer_train = cfg.get("qformer_train", False)

        sd_train_text_encoder = cfg.get("sd_train_text_encoder", False)
        sd_pretrained_model_name_or_path = cfg.get(
            "sd_pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"
        )

        controlnet_pretrained_model_name_or_path = cfg.get(
            "controlnet_pretrained_model_name_or_path", None
        )

        vae_half_precision = cfg.get("vae_half_precision", False)

        model = cls(
            vit_model=vit_model,
            qformer_cross_attention_freq=qformer_cross_attention_freq,
            qformer_num_query_token=qformer_num_query_token,
            qformer_train=qformer_train,
            sd_train_text_encoder=sd_train_text_encoder,
            sd_pretrained_model_name_or_path=sd_pretrained_model_name_or_path,
            controlnet_pretrained_model_name_or_path=controlnet_pretrained_model_name_or_path,
            vae_half_precision=vae_half_precision,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def load_checkpoint_from_dir(self, checkpoint_dir_or_url):
        # if checkpoint_dir is a url, download it and untar it
        if is_url(checkpoint_dir_or_url):
            checkpoint_dir_or_url = download_and_untar(checkpoint_dir_or_url)

        logging.info(f"Loading pretrained model from {checkpoint_dir_or_url}")

        def load_state_dict(module, filename):
            try:
                state_dict = torch.load(
                    os.path.join(checkpoint_dir_or_url, filename), map_location="cpu"
                )
                msg = module.load_state_dict(state_dict, strict=False)
            except FileNotFoundError:
                logging.info("File not found, skip loading: {}".format(filename))

        load_state_dict(self.proj_layer, "proj_layer/proj_weight.pt")
        load_state_dict(self.blip, "blip_model/blip_weight.pt")
        load_state_dict(self.unet, "unet/diffusion_pytorch_model.bin")
        load_state_dict(self.vae, "vae/diffusion_pytorch_model.bin")
        load_state_dict(self.text_encoder, "text_encoder/pytorch_model.bin")

        try:
            self.ctx_embeddings_cache.data = torch.load(
                os.path.join(
                    checkpoint_dir_or_url, "ctx_embeddings_cache/ctx_embeddings_cache.pt"
                ),
                map_location=self.device,
            )
            self._use_embeddings_cache = True
            print("Loaded ctx_embeddings_cache from {}".format(checkpoint_dir_or_url))
        except FileNotFoundError:
            self._use_embeddings_cache = False
            print("No ctx_embeddings_cache found in {}".format(checkpoint_dir_or_url))

    def load_from_pretrained(self, url_or_filename):
        checkpoint_dir = url_or_filename
        self.load_checkpoint_from_dir(checkpoint_dir)

    def load_checkpoint(self, url_or_filename):
        """
        Used to load finetuned models.
        """

        super().load_checkpoint(url_or_filename)

        print("loading fine-tuned model from {}".format(url_or_filename))
        self._use_embeddings_cache = True


# Null-Text Inversion
        
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = np.array(image_path)[:, :, :3]
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:
    # changed
    def __init__(self, device, scheduler, tokenizer, text_encoder, vae, unet, noise_scheduler, prompt=None, context=None, num_ddim_steps=50, guidance_scale=7.5):
        self.device = device
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.prompt = prompt
        self.context = context
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.scheduler.set_timesteps(num_ddim_steps)

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(input_ids=uncond_input.input_ids.to(self.device), ctx_embeddings=None)[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(input_ids=text_input.input_ids.to(self.device), ctx_embeddings=None)[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        # register_attention_control(self.model, None)
        image_gt = load_512(image, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    @property
    def scheduler(self):
        return self._scheduler  # Assuming you store the scheduler in a _scheduler attribute

    @scheduler.setter
    def scheduler(self, new_scheduler):
        self._scheduler = new_scheduler

# Negative-Prompt Inversion

# scheduler

class NegativePromptInversion:
    def __init__(self, device, scheduler, tokenizer, text_encoder, vae, unet, noise_scheduler, prompt=None, context=None, num_ddim_steps=50, guidance_scale=7.5):
        self.device = device
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.prompt = prompt
        self.context = context
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale
        self.scheduler.set_timesteps(num_ddim_steps)
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(input_ids=uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(input_ids=text_input.input_ids.to(self.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents, latent

    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), npi_interp=0.0, verbose=False):
        self.init_prompt(prompt)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents, image_rec_latent = self.ddim_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0:
            cond_embeddings = slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return (image_gt, image_rec, image_rec_latent), ddim_latents[-1], uncond_embeddings
