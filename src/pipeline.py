"""Pipeline used to create a stable diffusion dataset from a set of initial prompts."""
from pathlib import Path

import pyarrow as pa
from fondant.pipeline import Pipeline

from components.generate_prompts import GeneratePromptsComponent


BASE_PATH = "./data_dir"
# Create data directory if it doesn't exist
Path(BASE_PATH).mkdir(parents=True, exist_ok=True)

# Create your pipeline
pipeline = Pipeline(
    name="controlnet-pipeline",
    description="Pipeline that collects data to train ControlNet",
    base_path=BASE_PATH,
)

prompts = pipeline.read(
    GeneratePromptsComponent,
    arguments={
        "n_rows_to_load": 10
    },  # Set to 10 for small scale testing, set to None to load all rows
)

image_urls = prompts.apply(
    "retrieve_from_faiss_by_prompt",
    arguments={
        "url_mapping_path":"hf://datasets/fondant-ai/datacomp-small-clip/id_mapping",
        "faiss_index_path":"hf://datasets/fondant-ai/datacomp-small-clip/faiss",
        "num_images": 2
    },
)

images = image_urls.apply(
    "download_images",
    arguments={
        "timeout": 1,
        "retries": 0,
        "image_size": 512,
        "resize_mode": "center_crop",
        "resize_only_if_bigger": False,
        "min_image_size": 0,
        "max_aspect_ratio": 2.5,
    },
)

captions = images.apply(
    "caption_images",
    arguments={
        "model_id": "Salesforce/blip-image-captioning-base",
        "batch_size": 8,
        "max_new_tokens": 50,
    },
)

segmentations = captions.apply(
    "segment_images",
    arguments={
        "model_id": "openmmlab/upernet-convnext-small",
        "batch_size": 8,
    },
)


# OPTIONAL: writing the dataset to the Hugging Face Hub
HF_USER = None  # Insert your huggingface username here
HF_TOKEN = None  # Insert your HuggingFace token here

if HF_USER and HF_TOKEN:
    segmentations.write(
        "write_to_hf_hub",
        arguments={
            "username": HF_USER,
            "dataset_name": "fondant-controlnet-dataset",
            "hf_token": HF_TOKEN,
            "image_column_names": ["image"],
        },
        consumes={
            "image": pa.binary(),
            "image_width": pa.int32(),
            "image_height": pa.int32(),
        },
    )
