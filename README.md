# Lipschitz-constrained neural networks
Master semester project carried out at the Biomedical imaging group (BIG) at EPFL during the spring 2021 semester under the supervision of Bohra Pakshal

#### Description
Plug-and-Play methods, a subclass of variational methods in inverse problem, is characterized by a particularly modular structure which allows to plug-in state-of-the-art image denoisers, instead of optimization derived denoisers. The goal is to investigate the convergence's properties of Convolutional Neural Networks denoisers and their performance when plug-in in such frameworks. This work aims to extend the previous work  done on the subject by **[Ryu et al](https://github.com/uclaopt/Provable_Plug_and_Play)**. The goal is to build firmly  nonexpansive CNNs to soften assumption's made by Ryu et al. on the  data-fidelity term to broaden the theoretical convergence of such algorithm in the image reconstruction field, while reaching approximately the same performance level. On the road, we will study lipschitz-constrained CNNs and B-spline activations.  
#### Requirements

The required packages are `pytorch` and `torchvision`,  `opencv` and `h5py` for data-preprocessing, `cvxpy`, `cvxpylayers` and `qpth` for quadratic programming and `tqdm` for showing the training progress.
With some additional modules like `dominate` to save the results in the form of HTML files. To setup the necessary modules, simply run:

```bash
pip install -r requirements.txt
```

#### Datasets
In this project, we use two datasets : NYU fastMRI nad BSD500.

In the first part, we use **[fastMRI challenge](https://fastmri.org/)**, to obtain it click on the link and ask for access.
Once obtained, save the folders containing the ```.h5``` files under the ```data/``` directory.

The dataloader in this framework needs to work with 2d slices as input, thus the first thing to do is to extract the 2d 
images from the MRI volume contained in the ```.h5``` files. In the project directory do the following command to 
extract the images.

```bash
1) cd data
2 ) python select_slices.py --volumedir singlecoil_train --outputdir singlecoil_train_5_2d --nbslices 5
```
Here are the flags available for extraction:

```
--volumedir      Directory containing the .h5 files
--outputdir      Directory where 2d images are saved
--nbslices       The number of slices to extract per volume (needs to be odd, extract around the center of the volume)
```
In the second part we use **[BSD500](https://github.com/uclaopt/Provable_Plug_and_Play/tree/master/training/data)**, which can be downladed by clicking on the link and downloading the data folder from the github repository.
Save the three sub-folders ```train```, ```Set68``` amd ```Set12``` under the ```data``` folder from this repository.

To prepare the data do the following commands:
```bash
1) cd data
2 ) python BSD_preprocessing.py 
```
This will saved the augmented train and validation in two ```.h5``` files under the folder ```train_BSD500_preprocessed``` and ```val_BSD500_preprocessed```.

#### Training

To train a model, first download fastMRI or/and BSD500 as detailed above, then set `data_dir` to the dataset path in the config file in `configs/config.json` and set the rest of the parameters such as the seed used for initialization of the pytorch's model, etc..., you can also change the model's hyperparameters if you wish, more details below on the config section. Then simply run:

```bash
python train.py --config configs/config.json --device cpu or gpu
```

The log files and the `.pth` checkpoints will be saved in `saved\EXP_NAME\EXP_NAME_seed`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir directory of the tf.events files
```

To resume training using a saved `.pth` model:

```bash
python train.py --config directory_of _the_experiment/config.json --resume directory_of _the_experiment/checkpoint.pth
```

**Results**: The results will be saved in `saved` as an html file, containing the validation results,
and the name it will take is `experim_name` specified in `configs/config.json`.

### Inference

For inference, we need a pretrained model, the png images we'd like to test (path in the config file) and the config used in training (to load the correct model and other parameters): 

```bash
python inference.py --config config.json --model checkpoint.pth --experiment experiment_folder
```

The predictions and corresponding targets, will be saved as `.png` images in the `test_result\` folder as well as the singular values, the maximum slopes of B-spline activation (if used ) ,the MSE loss value and the SNR in  `.txt`  and `.pt` files.

Here are the flags available for inference:

```
--experiment   Path to the folder where to save the folder containing the test results
--model        Path to the trained pth model.
--config       The config file used for training the model.
```

#### PnP experiment

To run the experiments, we need a pretrained model with its config file, an image to test on, a mask and the noise. To run the experiments do the following commands:
```bash
1) cd PnP
2 ) python pnp_admm_csmri.py or pnp_fbs_csmri.py --config path/config.json --model path/model.pth --img path/img.jpg --mask path/mask.mat --experiment experimentname --sigma 15 or 5
```
The results will be saved under the Experiments folder containing the admm and fbs folders depending on the algorithm ( ```.py``` file) chosen.️

Here are the flags available :
```
-config       The config file used for training the model.
--model       Path to the trained pth model.
--img         Path to the image either Brain.jpg or Bust.jp
--mask        Path to the mask Q_Random30.mat, Q_Radial30.mat or Q_Cartesian30.mat
--jpg         Boolean True if Brain or Bust, Flase if use fastMRI
--noise       Path tothe noise noises.mat
--device      Device to use
--experiment  Name of the experiment
--mu          Scaling parameter, don't change here
--sigma       Noise level to use 15 for admm and 5 for fbs
--alpha       Hyperparameter alpha 2.0 for admm and 0.4 for fbs
--maxitr      Number of iterations
--verbose      Boolean to print info during the run
```

The scaling experiment can be runned using this commands: 

```bash
1) cd PnP
2 ) python DenoiserScaling.py --config path/config.json --model path/model.pth --img path/img.jpg --mask path/mask.mat --experiment experimentname --algo admm or fbs --mu_upper upperbound --mu_lower lowerbound --mu_step step --sigma 15 or 5
```
Here are the flags available :
```
-config       The config file used for training the model.
--model       Path to the trained pth model.
--img         Path to the image either Brain.jpg or Bust.jp
--mask        Path to the mask Q_Random30.mat, Q_Radial30.mat or Q_Cartesian30.mat
--jpg         Boolean True if Brain or Bust, Flase if use fastMRI
--noise       Path tothe noise noises.mat
--device      Device to use
--experiment  Name of the experiment
--algo        Algorithm to use
--mu_upper    Upperbound of the range of mu values
--mu_lower    Lowerbound of the range of mu values
--mu_step     step of the mu values 
--sigma       Noise level to use 15 for admm and 5 for fbs
--alpha       Hyperparameter alpha 2.0 for admm and 0.4 for fbs
--maxitr      Number of iterations
--verbose      Boolean to print info during the run
```
#### Config file details️

Bellow we detail the model's parameters that can be controlled in the config file `configs/config.json`.

```javascript
{
    "name": "DnCNN",
    "experim_name": "DnCNN",                              // experiment name
    "seeds": [42],                                        // pytorch seeds
    "dataset": "fastMRI",                                 // dataset used
    "sigma": 0.05,                                        // noise level used

    "model":{
        "depth": 4,                                       // number of convolutional layer
        "n_channels": 64,                                 // number of features map
        "image_channels": 1,                              // number of channels of th input image ( 1 grayscale)
        "kernel_size": 3,                                 // kernel size 
        "padding": 1,                                     // padding
        "bias": true,                                     // boolean to add bias to convolutional layer
        "architecture": "halfaveraged",                   // mapping used either residual, halfaveraged or direct
        "spectral_norm": "Parseval",                      // Spectral nOrmalization used None, Normal (SN), Chen (RealSN) or Parseval
        "batchnorm": false,                               // Boolean to use batch norm
        "beta": 0.6,                                      // Parseval strength parameter
        "activation_type": "relu",                        // activation to use relu, deepBspline, deepBspline_lipschitz_maxprojection, deepBspline_lipschitz_orthoprojection
        "shared_activation": false,                       // wheter to share activation module in B-splines
        "shared_channels": false,                         // wheter to share activation map for convolutions in b-splines
        "QP": "cvxpy",                                    // library used in orthoprojection cvxpy or qpth
        "spline_init" : "relu",                           // spline activation
        "spline_size": 51,                                // number of knots in B-splines
        "spline_range": 0.1,                              // step size in B-splines
        "slope_diff_threshold": 0,                        // threshold on slopes 
        "sparsify_activations" : false,                   // wheter to sparsify activation based on threshold at the end of training
        "hyperparam_tuning" : false,                      // boolean to tune hyperparameters
        "lipschitz" : false,                              // boolean to compute lipschitz regularization (not used in this project) 
        "lmbda" : 1e-4,                                   // total variation  hyperparameter
        "outer_norm" : 1,                                 // outer norm used
        "weight_decay" : 5e-4                             // weight decay
    },


    "optimizer": {
        "type": ["Adam"],
        "args":{
            "lr": 1e-3,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "stepscheduler": true,
            "step": 25,
            "gamma": 0.1
        }
    },


    "train_loader": {
        "target_dir": "data/singlecoil_train_5_2_0.05/Target_tensors",
        "noise_dirs": ["data/singlecoil_train_5_2_0.05/Noise_tensors_1", "data/singlecoil_train_5_2_0.05/Noise_tensors_2"],
        "batch_size": 25,
        "shuffle": true,
        "num_workers": 0
    },


    "val_loader": {
        "target_dir": "data/singlecoil_validation_5_1_0.05/Target_tensors",
        "noise_dirs": ["data/singlecoil_validation_5_1_0.05/Noise_tensors_1"],
        "batch_size": 25,
        "shuffle": false,
        "num_workers": 0
    },

    "test_loader": {
        "target_dir": "data/singlecoil_test_5_1_0.05/Target_tensors",
        "noise_dirs": ["data/singlecoil_test_5_1_0.05/Noise_tensors_1"],
        "batch_size": 1,
        "shuffle": false,
        "num_workers": 0
    },

    "trainer": {
        "epochs": 600,
        "save_dir": "saved/",
        "save_period": 25,
        
        "tensorboardX": true,
        "log_dir": "saved/",
        "log_per_iter": 100,

        "val": true,
        "val_per_epochs": 25
    }
}
```

If the dataset used is ```BSD500```, there is no noise directories, just change the target directory in the config file.

To run a SimpleCNN:
```javascript
{
    "name": "SimpleCNN",
    "experim_name": "SimpleCNN",                              // experiment name
    "seeds": [42],                                        // pytorch seeds
    "dataset": "fastMRI",                                 // dataset used
    "sigma": 0.05,                                        // noise level used

    "model":{
        "depth": 4,                                       
        "n_channels": 64,                                 
        "image_channels": 1,                              
        "kernel_size": 3,                                 
        "padding": 1,                                                                                     
        "batchnorm": false,                               
    }
}
```

To run a DnCNN:
```javascript
{
    "name": "DnCNN",
    "experim_name": "DnCNN",                              // experiment name
    "seeds": [42],                                        // pytorch seeds
    "dataset": "fastMRI",                                 // dataset used
    "sigma": 0.05,                                        // noise level used

    "model":{
        "depth": 17,                                       
        "n_channels": 64,                                 
        "image_channels": 1,                              
        "kernel_size": 3,                                 
        "padding": 1,                                                                                     
        "batchnorm": true,                               
    }
}
```