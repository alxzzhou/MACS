# MACS

This repo contains the official implementation of
paper: [MACS: Multi-source Audio-to-image Generation with Contextual Significance and Semantic Alignment](https://arxiv.org/abs/2503.10287)
. MACS is the first model that explicitly separates multi-source audio to capture the rich audio components before
audio-to-image generation.

![Static Badge](https://img.shields.io/badge/arXiv-2503.10287-red?link=arxiv.org%2Fabs%2F2503.10287)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/alxzzz/MACS)

![](figs/model%20architecture%20new.jpg)

## Getting Started

The code was developed on Ubuntu 22.04 with CUDA 12.1 and PyTorch 2.3.0.

1. **(Optional)** Download pretrained weights of Stable Diffusion 1.4 and specify the path in `train.sh`
   and `inference.sh` (you may also use huggingface path: `CompVis/stable-diffusion-v1-4`).
2. Download pretrained weights of separator and MACS (UNet and MLP) [here](https://huggingface.co/alxzzz/MACS) and
   specify the path in `train.sh` and `inference.sh`.

## Download Datasets
| **FSD50K**                                     | **AudioSet-Eval**                                                                                    | **LLP-Multi**                                                  | **Landscape**                                                   |
|------------------------------------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------------------|
| [download](https://zenodo.org/records/4060432) | [download](http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset_eval_strong.tsv) | refer to the [link](https://github.com/YapengTian/AVVP-ECCV20) | refer to the [link](https://github.com/researchmm/MM-Diffusion) |

You may need to download **AudioSet-Eval** and **LLP-Multi** from CSV files, as these datasets are not provided directly. 
Please refer to the scripts in the `download/` directory. Additionally, you may find these download scripts useful: [link](https://github.com/search?q=download%20audioset&type=repositories).

## Preprocessing Datasets  

### AudioSet-Eval  
The file `asset/audioset_eval_frames.json` contains frames that capture the highest number of audio events in each video. For example, the entry `"ABC": [1, 2]` indicates that frames 1 and 2 in video `ABC` contain the most audio events.
You may either randomly select frames from this set or leverage models like CLIP to identify the most relevant frames.  

### LLP-Eval  
The files `asset/LLP_*.json` contain audio-visual coexistence data extracted using the model proposed in [AVVP](https://arxiv.org/pdf/2007.10558).
Each 10-second video in LLP is divided into 1-second clips, where the value $n_i$ at position $i$ represents the number of audio-visual coexistences in the $i$-th second. 
These files serve as references for frame extraction, and you may further utilize CLIP to identify the most informative frames.