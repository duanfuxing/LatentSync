import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import List, Tuple

class FrameInterpolator:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() // 2)
        self.chunk_size = 100
        
    def interpolate_frames(self, 
                          video_frames: np.ndarray,
                          audio_duration: float,
                          target_fps: int) -> np.ndarray:
        """优化的帧插值，避免虚影"""
        n_frames = len(video_frames)
        required_frames = int(audio_duration * target_fps)
        
        # 如果需要的帧数较少，直接抽取
        if required_frames <= n_frames:
            indices = np.linspace(0, n_frames-1, required_frames, dtype=int)
            return video_frames[indices]
        
        frames_to_insert = required_frames - n_frames
        total_gaps = n_frames - 1
        frames_per_gap = frames_to_insert / total_gaps
        
        # 创建优化的光流计算器
        flow = cv2.FarnebackOpticalFlow_create(
            numLevels=3,      # 金字塔层数
            pyrScale=0.5,     # 金字塔缩放比例
            winSize=15,       # 窗口大小
            numIters=3,       # 迭代次数
            polyN=5,          # 多项式展开阶数
            polySigma=1.2,    # 高斯权重标准差
            flags=0
        )
        
        result_frames = []
        
        def process_frame_pair(i):
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            
            # 计算当前间隙需要插入的帧数
            insert_count = int(round((i + 1) * frames_per_gap) - round(i * frames_per_gap))
            if insert_count == 0:
                return [frame1]
            
            # 计算双向光流
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 正向光流
            flow_forward = flow.calc(gray1, gray2, None)
            # 反向光流
            flow_backward = flow.calc(gray2, gray1, None)
            
            frames = [frame1]
            h, w = frame1.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            
            for t in range(1, insert_count + 1):
                t_factor = t / (insert_count + 1)
                
                # 计算正向位移
                forward_displacement = flow_forward * t_factor
                forward_x = x_coords + forward_displacement[..., 0]
                forward_y = y_coords + forward_displacement[..., 1]
                
                # 计算反向位移
                backward_displacement = flow_backward * (1 - t_factor)
                backward_x = x_coords + backward_displacement[..., 0]
                backward_y = y_coords + backward_displacement[..., 1]
                
                # 双向插值
                forward_warped = cv2.remap(frame1, forward_x, forward_y, 
                                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                backward_warped = cv2.remap(frame2, backward_x, backward_y,
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                
                # 计算权重
                forward_weight = 1 - t_factor
                backward_weight = t_factor
                
                # 加权混合
                interpolated = cv2.addWeighted(forward_warped, forward_weight,
                                             backward_warped, backward_weight, 0)
                
                # 应用遮罩以减少虚影
                diff = cv2.absdiff(forward_warped, backward_warped)
                mask = (diff.mean(axis=2) < 30).astype(np.float32)  # 阈值可调
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                # 基于遮罩选择像素
                final_frame = np.where(mask[..., np.newaxis] > 0.5, 
                                     interpolated,
                                     forward_warped if forward_weight > backward_weight else backward_warped)
                
                frames.append(final_frame.astype(np.uint8))
            
            return frames
        
        # 并行处理帧对
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for i in range(0, n_frames - 1, self.chunk_size):
                chunk_end = min(i + self.chunk_size, n_frames - 1)
                futures = []
                
                for j in range(i, chunk_end):
                    futures.append(executor.submit(process_frame_pair, j))
                
                # 收集结果
                for future in futures:
                    result_frames.extend(future.result())
        
        # 添加最后一帧
        result_frames.append(video_frames[-1])
        
        # 确保帧数精确匹配
        if len(result_frames) > required_frames:
            result_frames = result_frames[:required_frames]
        elif len(result_frames) < required_frames:
            while len(result_frames) < required_frames:
                result_frames.append(result_frames[-1])
        
        return np.array(result_frames)
    
    def process_video(self, 
                     video_frames: np.ndarray,
                     audio_duration: float,
                     target_fps: int) -> np.ndarray:
        """处理视频帧"""
        print(f"开始处理视频帧，使用 {self.num_workers} 个CPU线程")
        print(f"输入帧数: {len(video_frames)}")
        print(f"目标帧数: {int(audio_duration * target_fps)}")
        
        # 直接进行插值，不做平滑处理
        print("正在进行帧插值...")
        interpolated = self.interpolate_frames(video_frames, audio_duration, target_fps)
        
        print(f"最终帧数: {len(interpolated)}")
        print("处理完成")
        
        return interpolated