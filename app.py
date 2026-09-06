import gradio as gr
from backend.main import process_video

with gr.Blocks(title="Kicklytics - Football Analysis") as demo:

    gr.Markdown(
        """
        # ⚽ Kicklytics Player Detection

        Upload a football match video clip to get:

        - **Player detection** with team classification
        - **Ball tracking** with possession analysis
        - **Pitch graph visualization**
        - **Passing lane analysis**
        - **Match statistics**

        *Supports MP4, AVI, MOV formats*
        """
            )

    with gr.Row():

        with gr.Column(scale=1):

            input_video = gr.Video(
                label="📹 Upload Match Video",
                sources=["upload"],
                format="mp4"
            )

            process_btn = gr.Button(
                "🚀 Analyze Video",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=2):

            output_video = gr.Video(
                label="🎬 Processed Video",
                interactive=False
            )

    stats_output = gr.Markdown(
        label="📊 Statistics"
    )

    gr.Markdown(
        """
---

**How it works:** The model detects players, assigns them to teams based
on jersey colors, tracks the ball, calculates possession, projects players
onto the football pitch, and analyzes passing lanes.
"""
    )

    process_btn.click(
        fn=process_video,
        inputs=input_video,
        outputs=[
            output_video,
            stats_output
        ]
    )


if __name__ == "__main__":
    demo.launch()