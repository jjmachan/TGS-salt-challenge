# TGS Salt Challenge
My solution for the [TGS Salt Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge/overview) 
on Kaggle - **top 20%**

<p align="center">
  <img src="https://github.com/jjmachan/TGS-salt-challenge/blob/master/tgs_salt_banner.png" />
</p>

This challenge aims to build a model that can detect large salt deposits from image of seismic scans. Detecting salt 
is important beacause in most cases large accumulation of oil and gas are also found underneith these salt beds. 
Salt is also much easier to detech that oil and gas and so if a model can be trained to detect these salt beds 
it will be very profitable for the Oil and Gas companies because it makes oil discovery faster. 

## Setup
clone the repo and install the packages with `pip install -r requirements.txt`

At the time being I was using a custom build of the popular 
`Segmentation-Model-Pytorch` library so you have to install that
```
$ wget https://github.com/jjmachan/segmentation_models.pytorch/releases/download/0.1.3.3/segmentation_models_pytorch-0.1.3-py3-none-any.whl -q
$ pip install segmentation_models_pytorch-0.1.3-py3-none-any.whl -qq
```

### Dataset
The dataset is downloaded from the [competition
page](https://www.kaggle.com/c/tgs-salt-identification-challenge/data). There is
an EDA notebook in `nbs/EDA.ipynb` for you to view and understand the data
better. 

I've build the Pytorch loaders and some utility function to get started with the
dataset quickly. Those can be found in the `data.py` and `utils.py` file.

I've used Albumentation library for augmenting the dataset but these can be
easily switched out. Check the blog for more details about the performance
boost augmentation gave. 

### Training
The `trainer.py` is a script that will run the trainer and save the trained
model into the `saved_models/` directory. You can modify hyperparameters in the
script directly to try stuff out. 

In addition to that I've also added 2 notebooks that I used for building this
repo. 
1. `nbs/Baseline.ipynb` - is the code to create a Baseline model that gives
   fairly good accuracy. It's a useful starting point if you want to get started
   on your own. 

2. `nbs/TGS Training.ipynb` - is similar to the `trainer.py` but for rapid
   development and prefer notebooks for that like me. 

Training is fairly fast and takes at max 1 hr.

### Inference 
To run the inference with the trained model simple call the `infer.py` script or
use the functions provided. This is generate the submission file in the required
formate. To run the evaluation head over to the kaggle website and submit to
competition. 

I've implemented my own version of Test Time Augmentation(TTA) to give an
additional boost to the model's performance.
