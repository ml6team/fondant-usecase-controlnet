# 🍫 Building a Controlnet pipeline for interior design with Fondant

This example demonstrates an end-to-end fondant pipeline to collect and process data for the fine-tuning of a [ControlNet](https://github.com/lllyasviel/ControlNet) model, focusing on images related to interior design.

## Pipeline overview

There are 5 components in total, these are:

1. [**Prompt Generation**](components/generate_prompts): This component generates a set of seed prompts using a rule-based approach that combines various rooms and styles together, like “a photo of a {room_type} in the style of {style_type}”. As input, it takes in a list of room types (bedroom, kitchen, laundry room, ..), a list of room styles (contemporary, minimalist, art deco, ...) and a list of prefixes (comfortable, luxurious, simple). These lists can be easily adapted to other domains. The output of this component is a list of seed prompts.

2. [**Image URL Retrieval**](https://github.com/ml6team/fondant/tree/main/components/prompt_based_laion_retrieval): This component retrieves images from the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset based on the seed prompts. The retrieval itself is done based on CLIP embeddings similarity between the prompt sentences and the captions in the LAION dataset. This component doesn’t return the actual images yet, only the URLs. The next component in the pipeline will then download these images.

3. [**Download Images**](https://github.com/ml6team/fondant/tree/main/components/download_images): This component downloads the actual images based on the URLs retrieved by the previous component. It takes in the URLs as input and returns the actual images, along with some metadata (like their height and width).

4. [**Add Captions**](https://github.com/ml6team/fondant/tree/main/components/caption_images): This component captions all images using [BLIP](https://huggingface.co/docs/transformers/model_doc/blip). This model takes in the image and generates a caption that describes the content of the image. This component takes in a Hugging Face model ID, so it can use any [Hugging Face Hub model](https://huggingface.co/models).

5. [**Add Segmentation Maps**](https://github.com/ml6team/fondant/tree/main/components/segment_images): This component segments the images using the [UPerNet](https://huggingface.co/docs/transformers/model_doc/upernet) model. Each segmentation map contains segments of 150 possible categories listed [here](https://huggingface.co/openmmlab/upernet-convnext-small/blob/main/config.json#L110).

## Environment

Please check that the following prerequisites are:
- A python version between 3.8 and 3.10 is installed on your system
  ```shell
  python --version
  ```
- Docker compose is installed on your system and the docker daemon is running
  ```shell
  docker compose version
  docker info
  ```
- A GPU is available (recommended, but not required)
  ```shell
  nvidia-smi
  ```
- Fondant is installed
  ```shell
  fondant
  ```

## Implementing the pipeline

The pipeline is implemented in [pipeline.py](pipeline.py). Please have a look at the file so you 
understand what is happening.

For more details on the pipeline creation, you can have a look at the 
[pipeline.ipynb](pipeline.ipynb) notebook which describes the process step by step.

## Running the pipeline

This pipeline will generate prompts, retrieve urls of matching images in the LAION dataset, download them 
and generate corresponding captions and segmentations. If you added the optional `write_to_hf_hub` 
component, it will write the resulting dataset to the HF hub.

Fondant provides different runners to run our pipeline.
Here we will use the local runner, which utilizes Docker compose under the hood.
For an overview of all runners, check the [Fondant documentation](https://fondant.ai/en/latest/pipeline/#running-a-pipeline).

The runner will first download the reusable components from the 
component hub. Afterwards, you will see the components execute one by one.

```shell
fondant run local pipeline.py
```

## Exploring the dataset

You can explore the dataset using the fondant explorer, this enables you to visualize your output dataset at each component step. Use the side panel on the left to browse through the steps and subsets.

```shell
fondant explore -b data_dir
```

## Creating your own dataset

To create your own dataset, you can update the generate_prompts component to generate prompts 
describing the images you want.

The component is implemented as a 
[lightweight component](https://fondant.ai/en/latest/components/lightweight_components/)
at [./components/generate_prompts/__init__.py](./components/generate_prompts/__init__.py).
You can update it to create your own prompts.

If you now re-run your pipeline, the new changes will be picked up and Fondant will automatically 
execute the component with the changes included.

```shell
fondant run local pipeline.py
```

If you restart the Explorer, you'll see that you can now select a second pipeline in the left panel 
and inspect your new dataset.

```shell
fondant explore -b data_dir
```

## Scaling up

If you're happy with your dataset, it's time to scale up. Check 
[our documentation](https://fondant.ai/en/latest/components/lightweight_components/) for 
more information about the available runners.
