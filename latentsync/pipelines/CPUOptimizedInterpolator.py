import numpy as np
import torch
import cv2
from typing import List, Tuple
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class CPUOptimizedInterpolator:
    def __init__(self, num_workers=None):
        # 默认使用 CPU 核心数的一半作为工作线程
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() // 2)
        # 批处理大小调整为较小值，适应 CPU 处理
        self.batch_size = 16
        self.chunk_size = 200  # 每次处理的帧数
        
    def interpolate_frames(self, 
                          video_frames: np.ndarray, 
                          target_fps: int, 
                          original_fps: int) -> np.ndarray:
        """使用 CPU 优化的光流法进行帧插值"""
        n_frames = len(video_frames)
        target_frames = int((n_frames / original_fps) * target_fps)
        
        # 创建光流计算器，使用 CPU 友好的参数
        flow = cv2.FarnebackOpticalFlow_create(
            numLevels=4,      # 降低金字塔层数
            pyrScale=0.5,     
            winSize=15,       # 减小窗口大小
            numIters=3,       # 减少迭代次数
            polyN=5,          
            polySigma=1.2,    
            flags=0
        )
        
        result_frames = []
        
        # 分块处理帧
        for chunk_start in range(0, n_frames - 1, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_frames - 1)
            chunk_frames = video_frames[chunk_start:chunk_end + 1]
            
            # 并行处理多个光流计算
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for i in range(len(chunk_frames) - 1):
                    futures.append(
                        executor.submit(
                            self._process_frame_pair,
                            chunk_frames[i],
                            chunk_frames[i + 1],
                            flow,
                            target_fps,
                            original_fps
                        )
                    )
                
                # 收集结果
                for future in futures:
                    frame_results = future.result()
                    result_frames.extend(frame_results)
            
            # 添加最后一帧（如果是最后一个chunk）
            if chunk_end == n_frames - 1:
                result_frames.append(video_frames[-1])
        
        return np.array(result_frames)
    
    def _process_frame_pair(self, frame1, frame2, flow, target_fps, original_fps):
        """处理一对相邻帧"""
        results = [frame1]
        
        # 计算光流
        flow_mat = flow.calc(
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
            None
        )
        
        # 确定插值帧数
        n_interp = max(1, int(target_fps / original_fps) - 1)
        
        if n_interp > 0:
            h, w = frame1.shape[:2]
            
            # 生成网格
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            
            for t in range(1, n_interp + 1):
                t_factor = t / (n_interp + 1)
                
                # 计算位移
                displacement = flow_mat * t_factor
                
                # 计算新坐标
                pos_x = x_coords + displacement[..., 0]
                pos_y = y_coords + displacement[..., 1]
                
                # 使用重映射进行插值
                interpolated = cv2.remap(
                    frame1,
                    pos_x,
                    pos_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                results.append(interpolated)
        
        return results
    
    def smooth_transitions(self, video_frames: np.ndarray, window_size: int = 3) -> np.ndarray:
        """CPU 优化的平滑过渡"""
        n_frames = len(video_frames)
        result = np.zeros_like(video_frames)
        
        # 创建高斯权重
        weights = np.exp(-np.linspace(-window_size, window_size, 2*window_size+1)**2 
                        / (2 * (window_size/2)**2))
        weights = weights / weights.sum()
        
        def smooth_frame(idx):
            """平滑单个帧"""
            start_idx = max(0, idx - window_size)
            end_idx = min(n_frames, idx + window_size + 1)
            
            # 调整权重以匹配实际窗口大小
            curr_weights = weights[max(0, window_size-idx):min(2*window_size+1, window_size+n_frames-idx)]
            curr_weights = curr_weights / curr_weights.sum()
            
            # 加权平均
            frame_window = video_frames[start_idx:end_idx]
            return np.average(frame_window, axis=0, weights=curr_weights)
        
        # 并行处理平滑操作
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(smooth_frame, i) for i in range(n_frames)]
            
            for i, future in enumerate(futures):
                result[i] = future.result()
        
        return result.astype(np.uint8)
    
    def process_video(self, 
                     video_frames: np.ndarray,
                     target_fps: int,
                     original_fps: int,
                     smooth_window: int = 3) -> np.ndarray:
        """完整的视频处理流程"""
        print(f"开始处理视频帧，使用 {self.num_workers} 个CPU线程")
        print(f"输入帧数: {len(video_frames)}")
        
        # 帧插值
        print("正在进行帧插值...")
        interpolated = self.interpolate_frames(video_frames, target_fps, original_fps)
        
        print(f"插值后帧数: {len(interpolated)}")
        
        # 平滑处理
        print("正在进行平滑处理...")
        smoothed = self.smooth_transitions(interpolated, smooth_window)
        
        print("处理完成")
        return smoothed