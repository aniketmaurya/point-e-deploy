import os
from functools import partial
from secrets import choice

import gradio as gr
import lightning as L
from lightning.app.components.serve import ServeGradio
from PIL import Image

image_size_choices = [256, 512, 1024]


class StableDiffusionUI(ServeGradio):
    inputs = [
        gr.inputs.Textbox(default="cat reading a book", label="Enter the text prompt"),
        gr.Slider(value=1, minimum=1, maximum=9, step=1, label="Number of images"),
        gr.Radio(value=512, choices=image_size_choices),
    ]
    outputs = gr.Gallery(type="pil")
    examples = [["golden puppy playing in a pool"], ["cat reading a book"]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self):
        import os

        import torch
        from diffusers import StableDiffusionPipeline

        access_token = os.environ.get("access_token")

        # make sure you're logged in with `huggingface-cli login`
        print("loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=access_token,
        )
        pipe = pipe.to("cuda")
        print("model loaded")
        return pipe

    def predict(self, prompt, num_images, image_size):
        from torch import autocast

        height, width = image_size, image_size
        prompt = [prompt] * int(num_images)
        with autocast("cuda"):
            images = self.model(prompt, height=height, width=width)["sample"]
        return images

    def run(self, *args, **kwargs):
        self.inputs[-1].style(item_container=True, container=True)

        if self._model is None:
            self._model = self.build_model()
        fn = partial(self.predict, *args, **kwargs)
        fn.__name__ = self.predict.__name__
        gr.Interface(
            fn=fn,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            title="Visualize your words",
        ).launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=self.enable_queue,
        )


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.model_demo = StableDiffusionUI(cloud_compute=L.CloudCompute("gpu"))

    def run(self):
        self.model_demo.run()

    def configure_layout(self):
        return [
            {"name": "Visualize your words", "content": self.model_demo},
            {
                "name": "About us",
                "content": "https://stability.ai/",
            },
        ]


if __name__ == "__main__":
    app = L.LightningApp(RootFlow())
