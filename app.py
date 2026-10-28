import os
import gradio as gr
from backend.main import process_video

SAMPLE_VIDEO_PATH = "./media/sample.mp4"


def load_sample_video():
    """'Upload from repo' button: fill the video box with the bundled clip."""
    if not os.path.exists(SAMPLE_VIDEO_PATH):
        gr.Warning(f"Sample video not found at {SAMPLE_VIDEO_PATH}")
        return None
    return SAMPLE_VIDEO_PATH

with gr.Blocks(
    title="Kicklytics - Football Analysis"
) as demo:

    # DESCRIPTION
    gr.Markdown(
        """
#Kicklytics Football analytics Powered by YOLO

Upload a football match video clip to get:

- **Player detection** with team classification
- **Ball tracking** with possession analysis
- **Pitch graph visualization**
- **Passing lane analysis**

### Output Layout

⬆️ **Top:** Player Detection Video

⬇️ **Bottom:** Pitch Graph Video

*Supports MP4, AVI, MOV formats*
"""
    )


    with gr.Row():
        with gr.Column(scale=1):
            pass

        with gr.Column(scale=4):
            gr.Image(
                value="./media/outImg.png",
                label="Example Output",
                interactive=False,
                #show_download_button=False
            )

        with gr.Column(scale=1):
            pass

    with gr.Row():

        with gr.Column(scale=1):
            input_video = gr.Video(
                label="📹 Upload Match Video",
                sources=["upload"],
                format="mp4"
            )

            repo_btn = gr.Button(
                "Load a sample video",
                variant="secondary"
            )

            process_btn = gr.Button(
                "Analyze Video",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=2):
            output_video = gr.Video(
                label="🎬 Processed Video",
                interactive=False
            )

    gr.Markdown("## ⚙️ How it works")
    with gr.Row():
        with gr.Column(scale=1):
            pass

        with gr.Column(scale=4):
            gr.Image(
                value='./media/designImg.png',
                label="Kicklytics System Architecture",
                interactive=False,
            )

        with gr.Column(scale=1):
            pass

    repo_btn.click(
        fn=load_sample_video,
        inputs=None,
        outputs=input_video
    )

    process_btn.click(
        fn=process_video,
        inputs=input_video,
        outputs=output_video
    )



if __name__ == "__main__":
    demo.launch()