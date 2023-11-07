"""Pipeline used to create a stable diffusion dataset from a set of initial prompts."""
import logging
import fsspec
from pathlib import Path

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

# General configs
HF_USER = None  # Insert your huggingface username here
HF_TOKEN = None  # Insert your HuggingFace token here
BASE_PATH = "./data_dir"
N_ROWS_TO_LOAD = 10  # Set to None to load all rows

# Create data directory if it doesn't exist and if it's a local path
if fsspec.core.url_to_fs(BASE_PATH)[0].protocol == ("file", "local"):
    Path(BASE_PATH).mkdir(parents=True, exist_ok=True)

pipeline = Pipeline(
    pipeline_name="controlnet-pipeline",
    pipeline_description="Pipeline that collects data to train ControlNet",
    base_path=BASE_PATH,
)

# Define component ops
generate_prompts_op = ComponentOp(
    component_dir="components/generate_prompts",
    arguments={"n_rows_to_load": N_ROWS_TO_LOAD},
)

laion_retrieval_op = ComponentOp.from_registry(
    name="prompt_based_laion_retrieval",
    arguments={
        "num_images": 2,
        "aesthetic_score": 9,
        "aesthetic_weight": 0.5,
    },
)
download_images_op = ComponentOp.from_registry(
    name="download_images",
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
caption_images_op = ComponentOp.from_registry(
    name="caption_images",
    arguments={
        "model_id": "Salesforce/blip-image-captioning-base",
        "batch_size": 8,
        "max_new_tokens": 50,
    },
    number_of_accelerators=1,
    accelerator_name="GPU",
)
segment_images_op = ComponentOp.from_registry(
    name="segment_images",
    arguments={
        "model_id": "openmmlab/upernet-convnext-small",
        "batch_size": 8,
    },
    number_of_accelerators=1,
    accelerator_name="GPU",
)

# Construct your pipeline
pipeline.add_op(generate_prompts_op)
pipeline.add_op(laion_retrieval_op, dependencies=generate_prompts_op)
pipeline.add_op(download_images_op, dependencies=laion_retrieval_op)
pipeline.add_op(caption_images_op, dependencies=download_images_op)
pipeline.add_op(segment_images_op, dependencies=caption_images_op)


# Add write to hub component if HF_USER and HF_TOKEN are set
if HF_USER and HF_TOKEN:
    write_to_hub_controlnet = ComponentOp(
        component_dir="components/write_to_hub_controlnet",
        arguments={
            "username": HF_USER,
            "dataset_name": "segmentation_kfp",
            "hf_token": HF_TOKEN,
            "image_column_names": ["images_data"],
        },
    )
    pipeline.add_op(write_to_hub_controlnet, dependencies=segment_images_op)
