
### What is ControlNet?
ControlNet is an image generation model developed by [Zhang etl a., 2023](https://arxiv.org/abs/2302.05543) that gives the user more control over the image generation process. It is based on the [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) model, which generates images based on text and an optional image. The ControlNet model adds a third input, a conditioning image, that can be used for specifying specific wanted elements in the generated image.

The Controlnet architecture is made so that it can be used for a wide variety of tasks and conditionings, such as segmentation maps, edge maps, scribbles, depth maps and more. A big benefit of Controlnet is that it can be trained with a relatively small dataset, since it reuses a lot of the weights from the Stable Diffusion model. This makes it very accessible for people with limited access to big amounts of compute and data.

Conditioning examples:

* Semantic segmentation maps
* Scribbles
* Depth maps
* Canny edge maps
* M-LSD Lines
* HED Boundaries
* Human Pose
* Normal Maps
* Anime Line Drawings
* Whatever you imagination can come up with!


Useful links:

* https://github.com/lllyasviel/ControlNet
* https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet
* https://arxiv.org/abs/2302.05543


### Multi-Modal dataset
In order to train a Controlnet model we need three inputs: an `image`, a `caption` and a `conditioning image`. When building this dataset, we typically begin with choosing a certain `domain of images`, e.g. interior design, faces, paintings, whatever you want. Next we need a textual description of the content of the image, also known as a `caption`. In some cases, the caption is already present in the dataset, but in other cases we need to generate it ourselves. Lastly, we need a `conditioning image`, which is an image that contains the elements that we want to see in the generated image. For example, if we want to generate a bedroom with a certain style, we could use a scribble of the bedroom layout as a conditioning image.

It's important that the dataset has enough quality images and captions and contains all three inputs for each sample.

![Multi modal dataset](docs/art/pipelines/interior_design/multi_modal_dataset.png)

### LAION-5B
When building your dataset, the images are the main component, since they are the starting point for getting captions and conditioning maps. One way of getting your dataset is by using a ready-to-go dataset, such as your own private dataset or a public dataset.

Where to find a ready-to-go image dataset:

* https://huggingface.co/docs/datasets/index
* https://pytorch.org/vision/stable/datasets.html
* https://www.kaggle.com/datasets


However, if you want some more specific data, this is not always possible. Luckily, [LAION](https://laion.ai/) has invested a lot of brain power and resources to open source some great tools and data such as [LAION-5B](https://laion.ai/blog/laion-5b/) and [clip-retrieval](https://github.com/rom1504/clip-retrieval). They built the LAION-5B dataset by scraping and filtering Common Crawl in a smart way (using CLIP and filters) and compiled it into a [FAISS](https://github.com/facebookresearch/faiss) Semantic Search index. This index can be used to retrieve images based on a visual and textual input, which results in an incredible powerful and efficient way of getting images for your dataset.

To explore the LAION-5B dataset you can use the [clip frontend website](https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn.laion.ai&index=laion5B-H-14&useMclip=false).

For retrieving images, you need to have a small set of textual descriptions or example images. The LAION-5B dataset will then retrieve the URLs of the most similar images based on the CLIP embeddings of the input. These URLs can then be used to download the actual images.


### How to use ControlNet
ControlNet is currently supported in multiple frameworks, such as PyTorch and JAX, by the [Diffusers](https://github.com/huggingface/diffusers) library from [Hugging Face](https://huggingface.co/docs/diffusers/index). The Diffusers library has built some awesome tools around Diffusion models in general, and supports all the functionality that you need to train and use a ControlNet model, such as inpainting, img2img, sampling schedulers, etc.

Another great repository is [this one](https://github.com/lllyasviel/ControlNet), that contains multiple training scripts and examples. This repository also contains models that are compatible with the well-known `Stable-Diffusion WebUI` from [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

### Examples
If you want to test ControlNet yourself, you can use the following examples:

* [Hugging Face - ControlNet for Interior Design](https://huggingface.co/spaces/ml6team/controlnet-interior-design)
* [AUTOMATIC1111 colab](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb)
