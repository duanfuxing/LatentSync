import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime

CONFIG_PATH = Path("configs/unet/second_stage.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
    fix_frames,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")  # Change the filename as needed

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed, fix_frames)

    try:
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path  # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int, fix_frames: bool
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--fix_frames", type=bool, default=True)

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--fix_frames",
            str(fix_frames),
        ]
    )


# Create Gradio interface
with gr.Blocks(title="LatentSync Video Processing") as demo:
    gr.Markdown(
        """
    # 基于ByteDance的LatentSync二次开发部署
    <div style="display:flex;column-gap:4px;">
        <a href="https://github.com/duanfuxing/LatentSync">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a> 
    </div>
    
    > 主要改进：
    > - 增加了视频循环播放适应长语音问题
    > - 修复视频跳帧平滑过度问题
    > - 优化了内存占用和处理速度

    ### 使用说明：
    - 硬件要求：显存建议 ≥24GB
    - 音频要求：采样率16000Hz，推荐使用 <a href="https://modelscope.cn/studios/iic/CosyVoice2-0.5B">CosyVoice2</a> 生成
    - 视频要求：
      - 分辨率：1080p
      - 帧率：30fps
      - 人脸要求：面部清晰、无遮挡
    """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="驱动音频", type="filepath")
            video_input = gr.Video(label="参考视频")

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.5,
                    value=1.5,
                    step=0.5,
                    label="引导尺度",
                )
                inference_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="推理步数")

            with gr.Row():
                seed = gr.Number(value=1247, label="随机种子", precision=0)
                fix_frames = gr.Checkbox(value=True, label="启用跳帧修复", info="使用光流法修复视频跳帧问题")

            process_btn = gr.Button("开始推理")

        with gr.Column():
            video_output = gr.Video(label="输出视频")

            gr.Examples(
                examples=[
                    ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                    ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                    ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                ],
                inputs=[video_input, audio_input],
            )

    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
            seed,
            fix_frames,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True, server_port=6006)
