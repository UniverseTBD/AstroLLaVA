# Based on liuhaotian/LLaVA-1.6

import sys
import os
import argparse
import time
import subprocess

import gradio as gr
import llava.serve.gradio_web_server as gws

# Execute the pip install command with additional options
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wheel', 'setuptools'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flash-attn', '--no-build-isolation', '-U'])


def start_controller():
    print("Starting the controller")
    controller_command = [
        sys.executable,
        "-m",
        "llava.serve.controller",
        "--host",
        "0.0.0.0",
        "--port",
        "10000",
    ]
    print("Controller Command:", controller_command)
    return subprocess.Popen(controller_command)


def start_worker(model_path: str, bits=16):
    print(f"Starting the model worker for the model {model_path}")
    model_name = model_path.strip("/").split("/")[-1]
    assert bits in [4, 8, 16], "It can be only loaded with 16-bit, 8-bit, and 4-bit."
    if bits != 16:
        model_name += f"-{bits}bit"
    model_name += "-lora"
    worker_command = [
        sys.executable,
        "-m",
        "llava.serve.model_worker",
        "--host",
        "0.0.0.0",
        "--controller",
        "http://localhost:10000",
        "--model-path",
        model_path,
        "--model-name",
        model_name,
        "--model-base",
        "liuhaotian/llava-1.5-7b",
        "--use-flash-attn",
    ]
    print("Worker Command:", worker_command)
    return subprocess.Popen(worker_command)


def handle_text_prompt(text, temperature=0.2, top_p=0.7, max_new_tokens=512):
    """
    Custom API endpoint to handle text prompts.
    Replace the placeholder logic with actual model inference.
    """
    # TODO: Replace the following placeholder with actual model inference code
    print(f"Received prompt: {text}")
    print(f"Parameters - Temperature: {temperature}, Top P: {top_p}, Max New Tokens: {max_new_tokens}")
    
    # Example response (replace with actual model response)
    response = f"Model response to '{text}' with temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}"
    return response


def add_text_with_image(text, image, mode):
    """
    Custom API endpoint to add text with an image.
    Replace the placeholder logic with actual processing.
    """
    # TODO: Replace the following placeholder with actual processing code
    print(f"Adding text: {text}")
    print(f"Image path: {image}")
    print(f"Image processing mode: {mode}")
    
    # Example response (replace with actual processing code)
    response = f"Added text '{text}' with image at '{image}' using mode '{mode}'."
    return response


def build_custom_demo(embed_mode=False, cur_dir='./', concurrency_count=5):
    """
    Builds a Gradio Blocks interface with custom API endpoints.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# AstroLLaVA")
        gr.Markdown("Welcome to the AstroLLaVA interface. Use the API endpoints to interact with the model.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Prompt the Model")
                text_input = gr.Textbox(label="Enter your text prompt", placeholder="Type your prompt here...")
                temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Top P")
                max_tokens_slider = gr.Slider(minimum=1, maximum=1024, value=512, step=1, label="Max New Tokens")
                submit_button = gr.Button("Submit Prompt")
            with gr.Column():
                chatbot_output = gr.Textbox(label="Model Response", interactive=False)

        submit_button.click(
            fn=handle_text_prompt,
            inputs=[text_input, temperature_slider, top_p_slider, max_tokens_slider],
            outputs=chatbot_output,
            api_name="prompt_model"  # Custom API endpoint name
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Add Text with Image")
                add_text_input = gr.Textbox(label="Add Text", placeholder="Enter text to add...")
                add_image_input = gr.Image(label="Upload Image")
                image_process_mode = gr.Radio(choices=["Crop", "Resize", "Pad", "Default"], value="Default", label="Image Process Mode")
                add_submit_button = gr.Button("Add Text with Image")
            with gr.Column():
                add_output = gr.Textbox(label="Add Text Response", interactive=False)

        add_submit_button.click(
            fn=add_text_with_image,
            inputs=[add_text_input, add_image_input, image_process_mode],
            outputs=add_output,
            api_name="add_text_with_image"  # Another custom API endpoint
        )

        # Additional API endpoints can be added here following the same structure

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AstroLLaVA Gradio App")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Hostname to listen on")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--controller-url", type=str, default="http://localhost:10000", help="Controller URL")
    parser.add_argument("--concurrency-count", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--model-list-mode", type=str, default="reload", choices=["once", "reload"], help="Model list mode")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly")
    parser.add_argument("--moderate", action="store_true", help="Enable moderation")
    parser.add_argument("--embed", action="store_true", help="Enable embed mode")
    args = parser.parse_args()
    gws.args = args
    gws.models = []

    gws.title_markdown += """ AstroLLaVA """

    print(f"AstroLLaVA arguments: {gws.args}")

    model_path = os.getenv("model", "universeTBD/AstroLLaVA_v2")
    bits = int(os.getenv("bits", 4))
    concurrency_count = int(os.getenv("concurrency_count", 5))

    controller_proc = start_controller()
    worker_proc = start_worker(model_path, bits=bits)

    # Wait for worker and controller to start
    print("Waiting for worker and controller to start...")
    time.sleep(30)

    exit_status = 0
    try:
        # Build the custom Gradio demo with additional API endpoints
        demo = build_custom_demo(embed_mode=False, cur_dir='./', concurrency_count=concurrency_count)
        print("Launching Gradio with custom API endpoints...")
        demo.queue(
            status_update_rate=10,
            api_open=False
        ).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        exit_status = 1
    finally:
        worker_proc.kill()
        controller_proc.kill()

        sys.exit(exit_status)
