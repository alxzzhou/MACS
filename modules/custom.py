import os.path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.schedulers import DDPMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from .separator import Separator


def init_parameters(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


# Modified from https://github.com/tencent-ailab/IP-Adapter
class IPAttnProcessor(nn.Module):
    def __init__(self, num_tokens, hidden_size, cross_attention_dim=None, scale=1.,
                 k_weight=None, v_weight=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        if k_weight is not None: self.to_k_ip.weight.data = k_weight.data
        if v_weight is not None: self.to_v_ip.weight.data = v_weight.data

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            # print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MLP(nn.Module):
    def __init__(self, M=6):
        super(MLP, self).__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(2048, 768),
            nn.LayerNorm(768)
        )
        self.pe = nn.Parameter(torch.zeros(M, 512))
        nn.init.normal_(self.pe, mean=0, std=0.05)

        self.apply(init_parameters)

    def forward(self, x):
        x = x + self.pe
        x = self.adapter(x)

        output = {}
        output['feat_vec'] = x
        return output


class StableDiffusion(nn.Module):
    def __init__(self,
                 sd_path,
                 pt_path,
                 train=True,
                 dataset=None,
                 M=6,
                 cfg=False):
        super(StableDiffusion, self).__init__()
        self.M = M
        self.cfg = cfg
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder='text_encoder')
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')
        self.set_adapter()

        if train:
            self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder='scheduler')
        else:
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_path, subfolder='scheduler')

        self.separator = Separator(M=M)
        weight = torch.load(os.path.join(pt_path, 'separator_FSD50K_full.pt'))
        self.separator.load_state_dict(weight)
        self.separator.load_clap()

        # MLP
        self.mlp = MLP(M=M)
        if not train:
            weight = torch.load(os.path.join(pt_path, f'{dataset}_mlp.pt'))
            self.mlp.load_state_dict(weight)

        # UNET
        if not train:
            weight = torch.load(os.path.join(pt_path, f'{dataset}_unet.pt'))
            self.unet.load_state_dict(weight)

        # grad setting
        if train:
            self.requires_grad_(False)
            self.eval()
            self.mlp.requires_grad_(True)
            self.mlp.train()
            for n, p in self.unet.named_modules():
                if isinstance(p, IPAttnProcessor):
                    p.requires_grad_(True)
                    p.train()
        else:
            self.requires_grad_(False)
            self.eval()

    def set_adapter(self):
        unet = self.unet
        state_dict = self.unet.state_dict()
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                k_weight = state_dict[name.replace('processor', 'to_k.weight')].detach().clone()
                v_weight = state_dict[name.replace('processor', 'to_v.weight')].detach().clone()
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=77,
                    scale=1.,
                    k_weight=k_weight, v_weight=v_weight
                ).to('cuda')
                print(f'[Adapter] {name} is set to IPAttnProcessor.')
        unet.set_attn_processor(attn_procs)
        print('[Adapter] IPAdapters are set.')

    def infer(self, audio, steps=20, device='cuda', prompt='', negative_prompt='', guidance_scale=7.5):
        size = 512
        hidden_coeff = self.separator(audio, None)['hidden_coeff']
        feat_vec = self.mlp(hidden_coeff)['feat_vec']

        input = self.tokenizer([prompt] * audio.shape[0], max_length=77, padding='max_length', truncation=True,
                               return_tensors="pt").input_ids.to('cuda')
        text_vec = self.text_encoder(input_ids=input).last_hidden_state  # [B, 77, 768]

        pos_hidden_states = torch.cat([text_vec, feat_vec], dim=1)

        latent = torch.randn(1, self.unet.in_channels, size // 8, size // 8,
                             device=device) * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(steps)

        if self.cfg:
            latent = torch.cat([latent] * 2, dim=0)

            hidden_coeff = torch.zeros_like(hidden_coeff).cuda()
            zero_input = self.mlp(hidden_coeff)['feat_vec']

            input = self.tokenizer([negative_prompt] * audio.shape[0], max_length=77, padding='max_length',
                                   truncation=True, return_tensors="pt").input_ids.to('cuda')
            text_vec = self.text_encoder(input_ids=input).last_hidden_state  # [B, 77, 768]

            zero_hidden_states = torch.cat([text_vec, zero_input], dim=1)
            hidden_states = torch.cat([pos_hidden_states, zero_hidden_states], dim=0)
            for ts in (self.scheduler.timesteps):
                t = torch.tensor([ts], dtype=torch.long, device=device)
                latent = self.scheduler.scale_model_input(latent, t)
                with torch.no_grad():
                    pos_output, zero_output = self.unet(latent, t, encoder_hidden_states=hidden_states).sample.chunk(2)
                output = zero_output + guidance_scale * (pos_output - zero_output)
                latent = self.scheduler.step(output, t, latent).prev_sample
        else:
            for ts in (self.scheduler.timesteps):
                t = torch.tensor([ts], dtype=torch.long, device=device)
                latent = self.scheduler.scale_model_input(latent, t)
                with torch.no_grad():
                    output = self.unet(latent, t, encoder_hidden_states=feat_vec).sample
                latent = self.scheduler.step(output, t, latent).prev_sample

        latent /= 0.18215
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        image = (image / 2 + .5).clamp(0, 1)  # [B, C, H, W]
        image = image.permute(0, 2, 3, 1)

        return image

    def forward(self, audio, image, prompt=''):
        mixit_output = self.separator(audio)
        hidden_coeff = mixit_output['hidden_coeff']

        if self.mlp.training and random.random() <= 0.1:
            hidden_coeff = torch.zeros_like(hidden_coeff).cuda()

        model_output = self.mlp(hidden_coeff)
        feat_vec = model_output.pop('feat_vec')  # [B, M, 768]

        # Text Encoder
        input = self.tokenizer([prompt] * audio.shape[0], max_length=77, padding='max_length', truncation=True,
                               return_tensors="pt").input_ids.to('cuda')
        text_vec = self.text_encoder(input_ids=input).last_hidden_state  # [B, 77, 768]

        # Stable Diffusion
        latent = self.vae.encode(image).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latent)
        encoder_hidden_states = torch.cat([text_vec, feat_vec], dim=1)

        bsz = latent.shape[0]
        time_step = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,),
                                  device=latent.device).long()
        noise_latent = self.scheduler.add_noise(latent, noise, time_step)
        unet_output = self.unet(noise_latent, time_step, encoder_hidden_states=encoder_hidden_states).sample

        loss = F.mse_loss(unet_output, noise, reduction="mean")

        output = {}
        output['sd_loss'] = loss
        output.update(mixit_output)
        output.update(model_output)
        return output
