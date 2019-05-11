# SALICONtf

This repository contains the code to train and run SALICONtf - the reimplementation of bottom-up saliency model SALICON in TensorFlow.

- [Implementation](#implementation)
- [Installation instructions](#installation)
- [Running SALICONtf](#running-salicontf-with-pretrained-weights)
- [Training SALICONtf](#finetuning-salicontf)

## Implementation

### Architecture
In our implementation we follow the original [CVPR'15 paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_SALICON_Reducing_the_ICCV_2015_paper.pdf) with several minor changes.

As in the original paper, SALICONtf model contains two VGG-based streams (without fc layers) for fine- and coarse-scale processing. Input is resized to 600x800px and 300x400px for fine and corase streams respectively. The final layer of the fine stream is resized to match the sie of the coarse stream (30x57px). Both outputs are concatenated and convolved with
1×1 filter. The labels (human fixation maps) are resized to 37×50 to match the output of the network.

### Training

In the original formulation the best results were achieved by optimizing the Kullback-Leibler divergence (KLD) loss. In our experiments with SALICONtf we obtained better results using the binary cross-entropy loss (which OpenSALICON also uses). We use fixed learning rate of 0.01 , momentum of 0.9 and weight decay of 0.0005. The original paper did not specify the number of training epochs and only mentioned that between 1 and 2 hours is required to train the model. Our implementation achieves reasonable results after 100 epochs and reaches its top perfomance on MIT1003 dataset after 300 epochs (which takes approx. 12 hours of training).


The model is trained on the OSIE dataset, which we split into training set of 630 images and validation set of 70 images. Batch size is set to 1. We evaluate the model on MIT1003 dataset. The results in the table below show that our model achieves results closest to the official SALICON demo results. The model runs at ≈ 5 FPS on the NVIDIA Titan X GPU.


Below are the results of our model compared to the [official SALICON demo](http://salicon.net/demo/) and [OpenSALICON](https://github.com/CLT29/OpenSALICON). We evaluate all models on [MIT1003 dataset](http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html) using [MIT benchmark code](https://github.com/cvzoya/saliency) to compute common metrics.

|                       |          |      | MIT1003 |      |      |
|-----------------------|----------|------|---------|------|------|
|         model         | AUC_Judd | CC   | KLDiv   | NSS  | SIM  |
| SALICON (online demo) | 0.87     | 0.62 | 0.96    | 2.17 | 0.5  |
| OpenSALICON           | 0.83     | 0.51 | 1.14    | 1.92 | 0.41 |
| SALICONtf             | 0.86     | 0.6  | 0.92    | 2.12 | 0.48 |


## Getting started
### Installation

We tested this setup with NVIDIA Titan X on Ubuntu 16.04 with Python 3.5.

SALICON needs about 5GB GPU memory, also make sure that you have a recent NVIDIA driver installed (version 384 or above).

#### Docker (stronlgy recommended)

Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) following the instructions in the official repository. There are also good resources elsewhere that describe Docker installation in more detail, for example [this one for Ubuntu 16.04](https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/).

After Docker is installed all you need to do is to build a container using the scripts in the ```docker_scripts``` folder:
```
sh docker_scripts/build
```

#### Without Docker
```
pip3 install -r requirements.txt
```

#### Download datasets and model weights

First download the pretrained weights for running the network and vgg weights for finetuning the network.

```
cd models
sh download_pretrained_weights.sh
sh download_vgg_weights.sh
```

Download OSIE dataset if you want to train SALICON. We provide fixation maps for the OSIE dataset which are generated from human fixation points (```osie_dataset/data/eye/fixations.mat```) using the MATLAB script ```generate_osie_fixation_maps.m```. 

```
cd osie_dataset
sh download_osie_dataset.sh
```

Download MIT1003 dataset used for evaluation (optional).
```
cd mit1003_dataset
sh download_mit1003.sh
```

### Running SALICONtf with pretrained weights
To run a pretrained SALICONtf on an arbitrary image directory use the docker script:
```
sh docker_scripts/run_batch 
```

Or without docker:
```
python3 src/run_SALICON.py -i <input_dir> -o <output_dir> [-w <model_weights>]
```

```input_dir``` and ```output_dir``` are the input and output directories respectively. If the output directory does not exist, it will be created. If no ```model_weights``` are provided, the pretrained model ```models/model_lr0.01_loss_crossentropy.h5``` will be used.


### Finetuning SALICONtf
To finetune SALICONtf on the original OSIE data using docker script:

```
sh docker_scripts/finetune
```

Or directly using the command:
```
python3 src/finetune_SALICON.py
```

<!-- ### Citing us

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{rasouli2017they,
  title={Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={206--213},
  year={2017}
}

@article{kotseruba2016joint,
  title={Joint attention in autonomous driving (JAAD)},
  author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
  journal={arXiv preprint arXiv:1609.04741},
  year={2016}
}
```
 -->
## Author

* **Yulia Kotseruba**

Please raise an issue or send email to yulia_k@cse.yorku.ca if there are any issues running the code.
