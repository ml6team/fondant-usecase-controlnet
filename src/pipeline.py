"""Pipeline used to create a stable diffusion dataset from a set of initial prompts."""
from pathlib import Path

from fondant.pipeline import Pipeline


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
    "components/generate_prompts",
    arguments={
        "n_rows_to_load": 10
    },  # Set to 10 for small scale testing, set to None to load all rows
)

laion_retrieval = prompts.apply(
    "retrieve_laion_by_prompt",
    arguments={
        "num_images": 2,
        "aesthetic_score": 9,
        "aesthetic_weight": 0.5,
    },
)

images = laion_retrieval.apply(
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
    segmentations.apply(
        "components/write_to_hub_controlnet",
        arguments={
            "username": HF_USER,
            "dataset_name": "fondant-controlnet-dataset",
            "hf_token": HF_TOKEN,
            "image_column_names": ["image"],
            "column_name_mapping": {
                "image": "image",
                "image_width": "width",
                "image_height": "height"
            }
        },
    )

