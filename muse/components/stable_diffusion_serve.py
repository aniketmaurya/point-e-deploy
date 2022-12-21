# !git clone https://github.com/openai/point-e.git
# !cd point-e && pip install -e . && cd ../

import base64
import matplotlib.pyplot as plt
import io
import os
import os.path
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import lightning as L  # noqa: E402
import torch  # noqa: E402
from lightning.app.storage import Drive  # noqa: E402
from PIL import Image  # noqa: E402

from muse.CONST import (  # noqa: E402
    IMAGE_SIZE,
    INFERENCE_REQUEST_TIMEOUT,
    KEEP_ALIVE_TIMEOUT,
)
from muse.utility.data_io import Data, DataBatch, TimeoutException  # noqa: E402


class StableDiffusionServe(L.LightningWork):
    """The StableDiffusionServer handles the prediction.

    It initializes a model and expose an API to handle incoming requests and generate predictions.
    """

    def __init__(
        self, safety_embeddings_drive: Optional[Drive] = None, safety_embeddings_filename: str = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.safety_embeddings_drive = safety_embeddings_drive
        self.safety_embeddings_filename = safety_embeddings_filename
        self._model = None
        self._trainer = None


    def build_pipeline(self):
        """The `build_pipeline(...)` method builds a model and trainer."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('creating base model...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        print('creating upsample model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        print('downloading base checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device))

        print('downloading upsampler checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))

        self._sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )
        print("loading model...")


    def fig_to_b64(self, fig):
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        return my_base64_jpgData


    def predict(self, prompts: List[Data], entry_time: int):
        samples = None
        prompts = [data.prompt for data in prompts]
        for x in tqdm(self._sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=prompts))):
            samples = x
        
        pc = self._sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
        return fig

    def run(self):

        import subprocess

        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        if torch.cuda.is_available():
            subprocess.run("nvidia-smi", shell=True)

        if self._model is None:
            self.build_pipeline()

        self._fastapi_app = app = FastAPI()
        app.POOL: ThreadPoolExecutor = None

        @app.on_event("startup")
        def startup_event():
            app.POOL = ThreadPoolExecutor(max_workers=1)

        @app.on_event("shutdown")
        def shutdown_event():
            app.POOL.shutdown(wait=False)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/api/health")
        def health():
            return True

        @app.post("/api/predict")
        def predict_api(data: DataBatch):
            """Dream a muse. Defines the REST API which takes the text prompt, number of images and image size in the
            request body.

            This API returns an image generated by the model in base64 format.
            """
            try:
                entry_time = time.time()
                print(f"batch size: {len(data.batch)}")
                result = app.POOL.submit(
                    self.predict,
                    data.batch,
                    entry_time=entry_time,
                ).result(timeout=INFERENCE_REQUEST_TIMEOUT)
                return self.fig_to_b64(result)
            except (TimeoutError, TimeoutException):
                raise TimeoutException()

        uvicorn.run(
            app, host=self.host, port=self.port, timeout_keep_alive=KEEP_ALIVE_TIMEOUT, access_log=False, loop="uvloop"
        )
