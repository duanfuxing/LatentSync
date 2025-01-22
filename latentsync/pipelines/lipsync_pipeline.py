# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf
from .CPUOptimizedInterpolator import CPUOptimizedInterpolator

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        # 初始化插值器
        self.interpolator = CPUOptimizedInterpolator()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, use_decord=False)
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, video_frames, boxes, affine_matrices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            # face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)
    
    # 在视频帧之间进行插值，平滑过渡
    def interpolate_frames(self, video_frames, target_fps, original_fps, batch_size=32):
        n_frames = len(video_frames)
        target_frames = int((n_frames / original_fps) * target_fps)
        
        with torch.cuda.amp.autocast():
            frames_tensor = torch.from_numpy(video_frames).cuda()
            
            # 分别处理每个通道
            channels = []
            for c in range(frames_tensor.shape[-1]):
                channel = frames_tensor[..., c]
                # 将通道重塑为 [batch_size, time] 形状
                channel = channel.reshape(-1, n_frames)  # [H*W, T]
                
                # 使用线性插值
                interpolated_channel = torch.nn.functional.interpolate(
                    channel.unsqueeze(0),  # [1, H*W, T]
                    size=target_frames,
                    mode='linear',
                    align_corners=False
                )
                
                # 恢复原始形状
                interpolated_channel = interpolated_channel.squeeze(0)  # [H*W, T]
                interpolated_channel = interpolated_channel.reshape(
                    frames_tensor.shape[0],  # H
                    frames_tensor.shape[1],  # W
                    target_frames
                )
                channels.append(interpolated_channel)
            
            # 合并所有通道
            interpolated = torch.stack(channels, dim=-1)  # [H, W, T, C]
            
        return interpolated.cpu().numpy()

    def smooth_transitions(self, video_frames, window_size=3):
        with torch.cuda.amp.autocast():
            frames = torch.from_numpy(video_frames).cuda().float()
            
            # 使用 3D 卷积代替逐通道处理
            kernel_size = window_size * 2 + 1
            gaussian_kernel = torch.exp(-torch.linspace(-window_size, window_size, kernel_size)**2 / (2 * (window_size/2)**2))
            gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).cuda()
            
            # 批量处理所有通道
            frames = frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
            padding = (0, 0, 0, 0, window_size, window_size)
            frames = torch.nn.functional.pad(frames, padding, mode='replicate')
            
            smoothed = torch.nn.functional.conv3d(
                frames,
                gaussian_kernel.view(1, 1, -1, 1, 1).expand(frames.size(1), -1, -1, 1, 1),
                groups=frames.size(1),
                padding=(0, 0, 0)
            )
            
            smoothed = smoothed.squeeze(0).permute(1, 2, 3, 0)
        
        return torch.clamp(smoothed, 0, 255).to(torch.uint8).cpu().numpy()

    def ensure_consistent_fps(self, video_frames, audio_samples, target_fps, audio_sample_rate):
        with torch.cuda.amp.autocast():
            audio_duration = len(audio_samples) / audio_sample_rate
            current_duration = len(video_frames) / target_fps
            
            print(f"输入视频帧维度: {video_frames.shape}")
            
            if abs(audio_duration - current_duration) > 0.1:
                required_frames = int(audio_duration * target_fps)
                
                # 更激进地降低分辨率
                scale_factor = 0.125  # 将分辨率降低到1/8
                h, w = int(video_frames.shape[1] * scale_factor), int(video_frames.shape[2] * scale_factor)
                
                # 更小的批处理大小
                batch_size = 8
                chunk_size = 4  # 每次处理的区域大小
                
                # 分块处理视频帧降采样
                resized_frames = []
                for i in range(0, len(video_frames), batch_size):
                    batch = video_frames[i:i+batch_size]
                    batch_resized = np.stack([
                        cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                        for frame in batch
                    ])
                    resized_frames.append(batch_resized)
                    torch.cuda.empty_cache()
                
                video_frames = np.concatenate(resized_frames, axis=0)
                
                if len(video_frames) < required_frames:
                    channels = []
                    
                    # 分通道处理
                    for c in range(video_frames.shape[-1]):
                        channel = video_frames[..., c]
                        channel_interpolated = np.zeros((required_frames, h, w), dtype=np.float32)
                        
                        # 分块处理每个通道
                        for y in range(0, h, chunk_size):
                            for x in range(0, w, chunk_size):
                                # 处理小块区域
                                y_end = min(y + chunk_size, h)
                                x_end = min(x + chunk_size, w)
                                
                                region = channel[:, y:y_end, x:x_end]
                                region_tensor = torch.from_numpy(region).cuda().float() / 255.0
                                
                                # 重塑并插值
                                region_flat = region_tensor.reshape(1, -1, region_tensor.shape[0])
                                interpolated = torch.nn.functional.interpolate(
                                    region_flat,
                                    size=required_frames,
                                    mode='linear',
                                    align_corners=False
                                )
                                
                                # 恢复形状并保存
                                interpolated = interpolated.reshape(
                                    required_frames, y_end-y, x_end-x
                                )
                                channel_interpolated[:, y:y_end, x:x_end] = interpolated.cpu().numpy()
                                
                                # 立即清理GPU内存
                                del region_tensor, region_flat, interpolated
                                torch.cuda.empty_cache()
                        
                        channels.append(channel_interpolated)
                        torch.cuda.empty_cache()
                    
                    # 合并通道
                    video_frames = np.stack(channels, axis=-1)
                    video_frames = (video_frames * 255.0).clip(0, 255).astype(np.uint8)
                    
                    # 分块恢复原始分辨率
                    final_height = video_frames.shape[1] * 8
                    final_width = video_frames.shape[2] * 8
                    final_frames = np.zeros((len(video_frames), final_height, final_width, 3), dtype=np.uint8)
                    
                    for i in range(0, len(video_frames), batch_size):
                        batch = video_frames[i:i+batch_size]
                        batch_upscaled = np.stack([
                            cv2.resize(frame, (final_width, final_height), 
                                     interpolation=cv2.INTER_LANCZOS4)
                            for frame in batch
                        ])
                        final_frames[i:i+batch_size] = batch_upscaled
                        torch.cuda.empty_cache()
                    
                    video_frames = final_frames
                else:
                    indices = torch.linspace(0, len(video_frames)-1, required_frames).long()
                    video_frames = video_frames[indices]
            
            print(f"输出视频帧维度: {video_frames.shape}")
        
        return video_frames

    # 修复视频跳帧问题-旧版本
    def fix_video_frames_old(self, video_frames, audio_samples, target_fps, audio_sample_rate):
        # 确保帧率一致性
        video_frames = self.ensure_consistent_fps(video_frames, audio_samples, target_fps, audio_sample_rate)
    
        # 使用GPU加速的平滑过渡
        video_frames = self.smooth_transitions(video_frames)
    
        # 确保帧数与音频长度匹配
        required_frames = int(len(audio_samples) / audio_sample_rate * target_fps)
        if len(video_frames) > required_frames:
            video_frames = video_frames[:required_frames]
    
        return video_frames

    # 修复视频跳帧问题
    def fix_video_frames(self, video_frames, audio_samples, target_fps, audio_sample_rate):
        # 计算目标持续时间
        audio_duration = len(audio_samples) / audio_sample_rate
        current_duration = len(video_frames) / target_fps
        
        if abs(audio_duration - current_duration) > 0.1:
            # 使用 CPU 优化的插值器处理
            video_frames = self.interpolator.process_video(
                video_frames,
                target_fps,
                len(video_frames) / current_duration,
                smooth_window=3
            )
        
        # 确保帧数与音频长度匹配
        required_frames = int(audio_duration * target_fps)
        if len(video_frames) > required_frames:
            video_frames = video_frames[:required_frames]
        
        return video_frames

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        fix_frames: bool = False,  # 跳帧修复
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        faces, original_video_frames, boxes, affine_matrices = self.affine_transform_video(video_path)
        audio_samples = read_audio(audio_path)

        video_frames = read_video(video_path, use_decord=False)
        original_video_frames = video_frames.copy()

        audio_duration = len(audio_samples) / audio_sample_rate
        video_duration = len(video_frames) / video_fps
        print(f"音频长度: {audio_duration} seconds")
        print(f"视频长度: {video_duration} seconds")

        # 视频倒放补帧
        if video_duration < audio_duration:
            repeat_factor = int(np.ceil(audio_duration / video_duration))
            print(f"视频长度小于音频长度，重复系数: {repeat_factor}")
            
            video_frames = np.tile(video_frames, (repeat_factor, 1, 1, 1))
            original_video_frames = np.tile(original_video_frames, (repeat_factor, 1, 1, 1))
            
            faces = torch.tile(faces, (repeat_factor, 1, 1, 1))
            boxes = boxes * repeat_factor
            affine_matrices = affine_matrices * repeat_factor
            print(f"扩展后的帧数:{video_frames.shape}")

        # 然后再进行帧数的裁剪
        required_frames = int(audio_duration * video_fps)
        video_frames = video_frames[:required_frames]
        original_video_frames = original_video_frames[:required_frames]
        faces = faces[:required_frames]
        boxes = boxes[:required_frames]
        affine_matrices = affine_matrices[:required_frames]

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

            num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
        else:
            num_inferences = len(faces) // num_frames

        synced_video_frames = []
        masked_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            batch_size,
            num_frames * num_inferences,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            image_latents = self.prepare_image_latents(
                pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)
            # masked_video_frames.append(masked_pixel_values)

        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), original_video_frames, boxes, affine_matrices
        )
        # masked_video_frames = self.restore_video(
        #     torch.cat(masked_video_frames), original_video_frames, boxes, affine_matrices
        # )

        # 处理视频跳帧
        if fix_frames:
            print("视频跳帧开始处理")
            synced_video_frames = self.fix_video_frames(
                video_frames=synced_video_frames,
                audio_samples=audio_samples,
                target_fps=self.video_fps,
                audio_sample_rate = audio_sample_rate
            )
            print(f"视频跳帧处理完成，最终帧数: {len(synced_video_frames)}")

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=25)
        # write_video(video_mask_path, masked_video_frames, fps=25)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
