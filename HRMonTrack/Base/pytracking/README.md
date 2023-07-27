# PyTracking
# change the model path in utils/loading.py
A general python library for visual tracking algorithms. 
## Table of Contents

* [Running a tracker](#running-a-tracker)
* [Overview](#overview)
* [Analysis](#analysis)
* [Libs](#libs)
* [Visdom](#visdom)
* [VOT Integration](#vot-integration)
* [Integrating a new tracker](#integrating-a-new-tracker)


## Running a tracker
The installation script will automatically generate a local configuration file  "evaluation/local.py". In case the file was not generated, run ```evaluation.environment.create_default_local_file()``` to generate it. Next, set the paths to the datasets you want
to use for evaluations. You can also change the path to the networks folder, and the path to the results folder, if you do not want to use the default paths. If all the dependencies have been correctly installed, you are set to run the trackers.  

The toolkit provides many ways to run a tracker.  

**Run the tracker on webcam feed**   
This is done using the run_webcam script. The arguments are the name of the tracker, and the name of the parameter file. You can select the object to track by drawing a bounding box. **Note:** It is possible to select multiple targets to track!
```bash
python run_webcam.py tracker_name parameter_name    
```  

**Run the tracker on some dataset sequence**  
This is done using the run_tracker script. 
```bash
python run_tracker.py tracker_name parameter_name --dataset_name dataset_name --sequence sequence --debug debug --threads threads
```  

Here, the dataset_name is the name of the dataset used for evaluation, e.g. ```otb```. See [evaluation.datasets.py](evaluation/datasets.py) for the list of datasets which are supported. The sequence can either be an integer denoting the index of the sequence in the dataset, or the name of the sequence, e.g. ```'Soccer'```.
The ```debug``` parameter can be used to control the level of debug visualizations. ```threads``` parameter can be used to run on multiple threads.

**Run the tracker on a set of datasets**  
This is done using the run_experiment script. To use this, first you need to create an experiment setting file in ```pytracking/experiments```. See [myexperiments.py](experiments/myexperiments.py) for reference. 
```bash
python run_experiment.py experiment_module experiment_name --dataset_name dataset_name --sequence sequence  --debug debug --threads threads
```  
Here, ```experiment_module```  is the name of the experiment setting file, e.g. ```myexperiments``` , and ``` experiment_name```  is the name of the experiment setting, e.g. ``` atom_nfs_uav``` .

**Run the tracker on a video file**  
This is done using the run_video script.  
```bash
python run_video.py experiment_module experiment_name videofile --optional_box optional_box --debug debug
```  
Here, ```videofile```  is the path to the video file. You can either draw the box by hand or provide it directly in the ```optional_box``` argument.

## Overview
The tookit consists of the following sub-modules.  
 - [analysis](analysis): Contains scripts to analyse tracking performance, e.g. obtain success plots, compute AUC score. It also contains a [script](analysis/playback_results.py) to playback saved results for debugging.
 - [evaluation](evaluation): Contains the necessary scripts for running a tracker on a dataset. It also contains integration of a number of standard tracking and video object segmentation datasets, namely  [OTB-100](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [NFS](http://ci2cv.net/nfs/index.html),
 [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx), [Temple128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [TrackingNet](https://tracking-net.org/), [GOT-10k](http://got-10k.aitestunion.com/), [LaSOT](https://cis.temple.edu/lasot/), [VOT](http://www.votchallenge.net), [Temple Color 128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [DAVIS](https://davischallenge.org), and [YouTube-VOS](https://youtube-vos.org).  
 - [experiments](experiments): The experiment setting files must be stored here,  
 - [features](features): Contains tools for feature extraction, data augmentation and wrapping networks.  
 - [libs](libs): Includes libraries for optimization, dcf, etc.  
 - [notebooks](notebooks) Jupyter notebooks to analyze tracker performance.
 - [parameter](parameter): Contains the parameter settings for different trackers.  
 - [tracker](tracker): Contains the implementations of different trackers.  
 - [util_scripts](util_scripts): Some util scripts for e.g. generating packed results for evaluation on GOT-10k and TrackingNet evaluation servers, downloading pre-computed results. 
 - [utils](utils): Some util functions. 
 - [VOT](VOT): VOT Integration.  
 
## Analysis  
The [analysis](analysis) module contains several scripts to analyze tracking performance on standard datasets. It can be used to obtain Precision and Success plots, compute AUC, OP, and Precision scores. The module includes utilities to perform per sequence analysis of the trackers. Further, it includes a [script](analysis/playback_results.py) to visualize pre-computed tracking results. Check [notebooks/analyze_results.ipynb](notebooks/analyze_results.ipynb) for examples on how to use the analysis module. 

## Libs
The pytracking repository includes some general libraries for implementing and developing different kinds of visual trackers, including deep learning based, optimization based and correlation filter based. The following libs are included:

* [**Optimization**](libs/optimization.py): Efficient optimizers aimed for online learning, including the Gauss-Newton and Conjugate Gradient based optimizer used in ATOM.
* [**Complex**](libs/complex.py): Complex tensors and operations for PyTorch, which can be used for DCF trackers.
* [**Fourier**](libs/fourier.py): Fourier tools and operations, which can be used for implementing DCF trackers.
* [**DCF**](libs/dcf.py): Some general tools for DCF trackers.

## Visdom

All trackers support [Visdom](https://github.com/facebookresearch/visdom) for debug visualizations. To use visdom, start the visdom
server from a seperate command line: 

```bash
visdom
```  

Run the tracker with the ```debug``` argument > 0. The debug output from the tracker can be 
accessed by going to ```http://localhost:8097``` in your browser. Further, you can pause the execution of the tracker,
or step through frames using keyboard inputs. 

![visdom](.figs/visdom.png)

## VOT Integration
#### Python Toolkit (VOT 2020)
Install the vot-python-toolkit and set up the workspace, as described in https://www.votchallenge.net/howto/tutorial_python.html. An example tracker description file to integrate a tracker in the vot-toolkit is provided at [VOT/trackers.ini](VOT/trackers.ini).


#### MATLAB Toolkit (VOT 2014-2019)
An example configuration file to integrate the trackers in the [VOT toolkit](https://github.com/votchallenge/vot-toolkit) is provided at [VOT/tracker_DiMP.m](VOT/tracker_DiMP.m). 
Copy the configuration file to your VOT workspace and set the paths in the configuration file. You need to install [TraX](https://github.com/votchallenge/trax) 
in order to run the trackers on VOT. This can be done with the following commands.

```bash
cd VOT_TOOLKIT_PATH/native/trax
mkdir build
cd build
cmake -DBUILD_OPENCV=ON -DBUILD_CLIENT=ON ..
make   
``` 

See https://trax.readthedocs.io/en/latest/index.html for more details about TraX.

## Integrating a new tracker  
 To implement a new tracker, create a new module in "tracker" folder with name your_tracker_name. This folder must contain the implementation of your tracker. Note that your tracker class must inherit from the base tracker class ```tracker.base.BaseTracker```.
 The "\_\_init\_\_.py" inside your tracker folder must contain the following lines,  
```python
from .tracker_file import TrackerClass

def get_tracker_class():
    return TrackerClass
```
Here, ```TrackerClass``` is the name of your tracker class. See the [file for DiMP](tracker/dimp/__init__.py) as reference.

Next, you need to create a folder "parameter/your_tracker_name", where the parameter settings for the tracker should be stored. The parameter fil shall contain a ```parameters()``` function that returns a ```TrackerParams``` struct. See the [default parameter file for DiMP](parameter/dimp/dimp50.py) as an example.

 
 
