import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import List, Tuple

class CPUOptimizedInterpolator:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() // 2)
        self.batch_size = 8
        self.chunk_size = 100
        
    def interpolate_frames(self, 
                          video_frames: np.ndarray,
                          audio_duration: float,  # 音频时长（秒）
                          target_fps: int) -> np.ndarray:
        """基于音频时长的精确帧插值"""
        n_frames = len(video_frames)
        # 根据音频时长计算所需的精确帧数
        required_frames = int(audio_duration * target_fps)
        
        # 如果所需帧数小于当前帧数，进行帧抽取
        if required_frames <= n_frames:
            indices = np.linspace(0, n_frames-1, required_frames, dtype=int)
            return video_frames[indices]
            
        # 计算实际需要插入的帧数
        frames_to_insert = required_frames - n_frames
        
        # 创建光流计算器
        flow = cv2.FarnebackOpticalFlow_create(
            numLevels=3,      # 金字塔层数
            pyrScale=0.5,     # 金字塔缩放比例
            winSize=15,       # 窗口大小
            numIters=3,       # 迭代次数
            polyN=5,          # 多项式展开阶数
            polySigma=1.2,    # 高斯权重标准差
            flags=0
        )
        
        # 计算每对帧之间需要插入的帧数
        total_gaps = n_frames - 1
        frames_per_gap = frames_to_insert / total_gaps
        
        result_frames = []
        
        def process_frame_pair(i):
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            
            # 计算当前间隙需要插入的帧数
            insert_count = int(round((i + 1) * frames_per_gap) - round(i * frames_per_gap))
            if insert_count == 0:
                return [frame1]
                
            # 计算光流
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow_mat = flow.calc(gray1, gray2, None)
            
            frames = [frame1]
            if insert_count > 0:
                h, w = frame1.shape[:2]
                y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
                
                for t in range(1, insert_count + 1):
                    t_factor = t / (insert_count + 1)
                    
                    # 计算位移
                    displacement = flow_mat * t_factor
                    pos_x = x_coords + displacement[..., 0]
                    pos_y = y_coords + displacement[..., 1]
                    
                    # 插值
                    interpolated = cv2.remap(
                        frame1,
                        pos_x,
                        pos_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    
                    frames.append(interpolated)
            
            return frames
        
        # 并行处理帧对
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(n_frames - 1):
                futures.append(executor.submit(process_frame_pair, i))
            
            # 收集结果
            for future in futures:
                result_frames.extend(future.result())
        
        # 添加最后一帧
        result_frames.append(video_frames[-1])
        
        # 确保帧数精确匹配
        if len(result_frames) > required_frames:
            result_frames = result_frames[:required_frames]
        elif len(result_frames) < required_frames:
            # 如果帧数不足，复制最后一帧
            while len(result_frames) < required_frames:
                result_frames.append(result_frames[-1])
        
        return np.array(result_frames)
    
    def smooth_transitions(self, video_frames: np.ndarray, window_size: int = 2) -> np.ndarray:
        """轻量级平滑处理"""
        n_frames = len(video_frames)
        result = np.zeros_like(video_frames)
        
        # 使用较小的高斯权重窗口
        weights = np.exp(-np.linspace(-window_size, window_size, 2*window_size+1)**2 
                        / (2 * (window_size/2)**2))
        weights = weights / weights.sum()
        
        def smooth_frame(idx):
            start_idx = max(0, idx - window_size)
            end_idx = min(n_frames, idx + window_size + 1)
            
            curr_weights = weights[max(0, window_size-idx):min(2*window_size+1, window_size+n_frames-idx)]
            curr_weights = curr_weights / curr_weights.sum()
            
            frame_window = video_frames[start_idx:end_idx]
            return np.average(frame_window, axis=0, weights=curr_weights)
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(smooth_frame, i) for i in range(n_frames)]
            
            for i, future in enumerate(futures):
                result[i] = future.result()
        
        return result.astype(np.uint8)
    
    def process_video(self, 
                     video_frames: np.ndarray,
                     audio_duration: float,
                     target_fps: int,
                     smooth_window: int = 2) -> np.ndarray:
        """完整的视频处理流程"""
        print(f"开始处理视频帧，使用 {self.num_workers} 个CPU线程")
        print(f"输入帧数: {len(video_frames)}")
        print(f"目标帧数: {int(audio_duration * target_fps)}")
        
        # 帧插值
        print("正在进行帧插值...")
        interpolated = self.interpolate_frames(video_frames, audio_duration, target_fps)
        
        print(f"插值后帧数: {len(interpolated)}")
        
        # 平滑处理（使用较小的窗口）
        print("正在进行平滑处理...")
        smoothed = self.smooth_transitions(interpolated, smooth_window)
        
        print("处理完成")
        return smoothed