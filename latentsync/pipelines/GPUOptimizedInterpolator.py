import numpy as np
import torch
import cv2
from typing import List, Tuple
import torch.nn.functional as F

class GPUOptimizedInterpolator:
    def __init__(self, device='cuda'):
        self.device = device
        # 针对A800优化的批处理大小
        self.batch_size = 32  # 可以处理更大的批次
        self.chunk_size = 1000  # 每次处理的帧数
        
    def interpolate_frames(self, 
                          video_frames: np.ndarray, 
                          target_fps: int, 
                          original_fps: int) -> np.ndarray:
        """使用GPU加速的光流法进行帧插值"""
        n_frames = len(video_frames)
        target_frames = int((n_frames / original_fps) * target_fps)
        
        # 创建优化的光流计算器
        flow = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=5,      # 金字塔层数
            pyrScale=0.5,     # 金字塔缩放比例
            winSize=21,       # 增大窗口大小，提高精度
            numIters=5,       # 增加迭代次数
            polyN=7,          # 多项式展开阶数
            polySigma=1.5,    # 高斯平滑参数
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN  # 使用高斯加权
        )
        
        # 预分配GPU内存
        gpu_frames = []
        for i in range(0, n_frames, self.chunk_size):
            chunk = video_frames[i:min(i + self.chunk_size, n_frames)]
            # 直接在GPU上分配内存
            gpu_chunk = torch.from_numpy(chunk).cuda()
            gpu_frames.append(gpu_chunk)
        
        result_frames = []
        
        for chunk_idx, gpu_chunk in enumerate(gpu_frames):
            chunk_frames = gpu_chunk.cpu().numpy()
            chunk_size = len(chunk_frames)
            
            # 并行处理多个光流计算
            for i in range(0, chunk_size - 1, self.batch_size):
                batch_end = min(i + self.batch_size, chunk_size - 1)
                
                # 批量计算光流
                flows = []
                for j in range(i, batch_end):
                    frame1 = cv2.cuda_GpuMat(cv2.cvtColor(chunk_frames[j], cv2.COLOR_BGR2GRAY))
                    frame2 = cv2.cuda_GpuMat(cv2.cvtColor(chunk_frames[j+1], cv2.COLOR_BGR2GRAY))
                    flow_gpu = cv2.cuda_GpuMat()
                    flow.calc(frame1, frame2, flow_gpu)
                    flows.append(flow_gpu.download())
                
                # 批量生成插值帧
                for j, flow_mat in enumerate(flows):
                    frame_idx = i + j
                    
                    # 确定插值帧数
                    n_interp = max(1, int(target_fps / original_fps) - 1)
                    
                    result_frames.append(chunk_frames[frame_idx])
                    
                    if n_interp > 0:
                        # 使用GPU进行插值计算
                        frame_tensor = torch.from_numpy(chunk_frames[frame_idx]).cuda()
                        next_frame_tensor = torch.from_numpy(chunk_frames[frame_idx + 1]).cuda()
                        flow_tensor = torch.from_numpy(flow_mat).cuda()
                        
                        for t in range(1, n_interp + 1):
                            t_factor = t / (n_interp + 1)
                            
                            # 使用光流进行插值
                            interpolated = self._interpolate_frame(
                                frame_tensor,
                                next_frame_tensor,
                                flow_tensor,
                                t_factor
                            )
                            
                            result_frames.append(interpolated.cpu().numpy())
                            
                        # 清理GPU内存
                        del frame_tensor, next_frame_tensor, flow_tensor
                
                torch.cuda.empty_cache()
            
            # 添加最后一帧
            if chunk_idx == len(gpu_frames) - 1:
                result_frames.append(chunk_frames[-1])
        
        return np.array(result_frames)
    
    def _interpolate_frame(self, frame1, frame2, flow, t_factor):
        """GPU加速的帧插值"""
        h, w = frame1.shape[:2]
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device)
        )
        
        # 计算位移
        flow_x = flow[..., 0] * t_factor
        flow_y = flow[..., 1] * t_factor
        
        # 计算采样位置
        pos_x = grid_x + flow_x
        pos_y = grid_y + flow_y
        
        # 归一化坐标
        pos_x = (pos_x / (w - 1) * 2 - 1).unsqueeze(0)
        pos_y = (pos_y / (h - 1) * 2 - 1).unsqueeze(0)
        
        # 构建采样网格
        grid = torch.stack([pos_x, pos_y], dim=-1)
        
        # 使用双线性插值
        frame1 = frame1.unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        interpolated = F.grid_sample(
            frame1,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return (interpolated[0].permute(1, 2, 0) * 255).to(torch.uint8)
    
    def smooth_transitions(self, video_frames: np.ndarray, window_size: int = 5) -> np.ndarray:
        """GPU加速的平滑过渡"""
        frames_tensor = torch.from_numpy(video_frames).cuda().float()
        
        # 使用3D卷积进行平滑
        kernel_size = 2 * window_size + 1
        gaussian_kernel = torch.exp(
            -torch.linspace(-window_size, window_size, kernel_size)**2 / 
            (2 * (window_size/2)**2)
        ).cuda()
        
        # 归一化权重
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        # 扩展维度以适应3D卷积
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)
        gaussian_kernel = gaussian_kernel.view(1, 1, -1, 1, 1).expand(3, -1, -1, 1, 1)
        
        # 添加padding
        padding = (0, 0, 0, 0, window_size, window_size)
        frames_tensor = F.pad(frames_tensor, padding, mode='replicate')
        
        # 应用平滑
        smoothed = F.conv3d(
            frames_tensor,
            gaussian_kernel,
            groups=3,
            padding=(0, 0, 0)
        )
        
        # 恢复维度顺序
        smoothed = smoothed.squeeze(0).permute(1, 2, 3, 0)
        
        return smoothed.clamp(0, 255).to(torch.uint8).cpu().numpy()

    def process_video(self, 
                     video_frames: np.ndarray,
                     target_fps: int,
                     original_fps: int,
                     smooth_window: int = 5) -> np.ndarray:
        """完整的视频处理流程"""
        # 帧插值
        interpolated = self.interpolate_frames(video_frames, target_fps, original_fps)
        
        # 平滑处理
        smoothed = self.smooth_transitions(interpolated, smooth_window)
        
        return smoothed