# SALICONtf

This repository contains the code to train and run SALICONtf - the reimplementation of bottom-up saliency model SALICON in TensorFlow.

## Implementation

In our implementation we follow the original [CVPR'15 paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_SALICON_Reducing_the_ICCV_2015_paper.pdf) with several minor changes.

As in the original paper, SALICONtf model contains two VGG-based streams (without fc layers) for fine- and coarse-scale processing. Input is resized to 600x800px and 300x400px for fine and corase streams respectively. The final layer of the fine stream is resized to match the sie of the coarse stream (30x57px). Both outputs are concatenated and convolved with
1×1 filter. The labels (human fixation maps) are resized to 37×50 to match the output of the network.

## Training

In the original formulation the best results were achieved by optimizing the Kullback-Leibler divergence (KLD) loss. In our experiments with SALICONtf we obtained better results using the binary cross-entropy loss as in [13]. We use fixed learning rate of 0.01 , momentum of 0.9 and weight decay of 0.0005. The original paper did not specify the number of training epochs and only mentioned that between 1 and 2 hours is required to train the model. Our implementation achieves reasonable results after 100 epochs and reaches its top perfomance on MIT1003 dataset after 300 epochs (which takes approx. 12 hours of training).


The model is trained on the OSIE dataset, which we split into training set of 630 images and validation set of 70 images. Batch size is set to 1. We evaluate the model on MIT1003 dataset. The results in the table below show that our model achieves results closest to the official SALICON demo results. The model runs at ≈ 5 FPS on the NVIDIA Titan X GPU.

|                       |          |      | MIT1003 |      |      |
|-----------------------|----------|------|---------|------|------|
|         model         | AUC_Judd | CC   | KLDiv   | NSS  | SIM  |
| SALICON (online demo) | 0.87     | 0.62 | 0.96    | 2.17 | 0.5  |
| OpenSALICON           | 0.83     | 0.51 | 1.14    | 1.92 | 0.41 |
| SALICONtf             | 0.86     | 0.6  | 0.92    | 2.12 | 0.48 |





```
xml
<?xml version="1.0" encoding="utf-8"?>
<video FPS="29.97" filename="video_0001.mp4" id="video_0001" length_sec="20.02" num_frames="600">
   <tags>
      <time_of_day val="daytime"/>
      <weather val="cloudy"/>
      <location val="plaza"/>
      <road_condition val="dry"/>
   </tags>
   <subjects>
      <Driver/>
      <pedestrian1/>
      <pedestrian2/>
   </subjects>
   <actions>
      <Driver>
         <action end_frame="57" end_time="1.9019" id="moving slow" start_frame="1" start_time="0"/>
         <action end_frame="141" end_time="4.7047" id="decelerating" start_frame="58" start_time="1.9353"/>
      </Driver>
      <pedestrian1>
         <action end_frame="364" end_time="12.133" id="standing" start_frame="1" start_time="0.02"/>
         <action end_frame="473" end_time="15.773" id="looking" start_frame="444" start_time="14.8"/>
      </pedestrian1>
      <pedestrian2>
         <action end_frame="70" end_time="2.336" id="walking" start_frame="1" start_time="0.02"/>
      </pedestrian2>
   </actions>
</video>
```

Xml files can be read in MATLAB using xml2struct.m script available at (https://www.mathworks.com/matlabcentral/fileexchange/28518-xml2struct)


## Pedestrian behavior attributes
Behavior attributes for each pedestrian are provided as a text file (pedestrian_attributes.txt).  

Each line lists attributes (comma-separated) for a single pedestrian in the following order:  
video_id, pedestrian_id, group_size, direction, designated, signalized, gender, age, num_lanes, traffic direction, intersection, crossing

* video_id, pedestrian_id, gender (male/female and n/a for small children) and age (child/young/adult/senior) are self-explanatory
* group_size: size of the group that the pedestrian is part of (moving or standing together)
* direction: indicates whether the pedestrian is moving along the direction of car's movement (LONG), crossing in front of the car (LAT) or standing (n/a)
* designated: the location where the pedestrian is moving/standing is designated for crossing (D) or non-designated (ND)
* signalized: the location where the pedestrian is moving/standing is signalized (S), i.e. has a stop sign or traffic lights, or not signalized (NS)
* num_lanes: number of lanes at the place where the pedestrian is moving/standing
* traffic direction: OW - one way, TW - two way
* intersection: yes - crossing at the intersection and no otherwise
* crossing: 1 - pedestrian completes crossing, 0 - pedestrian does not cross, -1 - no intention of crossing (e.g. waiting at the bus stop, talking to somebody at the curb)

When there are no pedestrians in the video, all attributes are set to "n/a".

## Traffic scene elements

We provide traffic_scene_elements.txt file  which lists scene elements for each video with corresponding frame numbers.  
The text is formatted as follows:  
video_id, attr_id: start_frame-end_frame; attr_id: start_frame-end_frame;  
Note: if no range is provided, the scene element is visible in all frames of the video.  

e.g. ```video_0005, 1: 1-30; 1: 90-240; 7;```

translates to the following : stop sign (1) is visible in frames 1-30 and 90-240, the entire video is filmed in a parking lot (7).

### Citing us

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

## Authors

* **Amir Rasouli**
* **Yulia Kotseruba**

Please send email to yulia_k@cse.yorku.ca or aras@cse.yorku.ca if there are any problems with downloading or using the data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
