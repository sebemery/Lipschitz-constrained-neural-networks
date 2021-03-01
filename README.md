# Lipschitz-constrained neural networks
Master semester project carried out at the Biomedical imaging group (LIB) at EPFL during the spring 2021 semester under the supervision of Bohra Pakshal

####Description
To Do

#### Requirements

The required packages are `pytorch` and `torchvision`,  `opencv` for data-preprocessing and `tqdm` for showing the training progress.
With some additional modules like `dominate` to save the results in the form of HTML files. To setup the necessary modules, simply run:

```bash
pip install -r requirements.txt
```

#### Dataset

In this repo, we use **[fastMRI challenge](https://fastmri.org/)**, to obtain it click on the link and ask for access.
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

#### Training

To train a model, first download fastMRI as detailed above, then set `data_dir` to the dataset path in the config file in `configs/config.json` and set the rest of the parameters such as the seed used for initialization of the pytorch's model, etc..., you can also change the model's hyperparameters if you wish, more details below. Then simply run:

```bash
python train.py --config configs/config.json
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

For inference, we need a pretrained model, the png images we'd like to test and the config used in training (to load the correct model and other parameters), 

```bash
python inference.py --config config.json --model checkpoint.pth --images images_folder
```

The predictions and corresponding targets will be saved as `.png` images in the `test_result\` folder as well as the MSE loss value in a `.txt` file.

Here are the flags available for inference:

```
--images       Folder containing the png images to segment.
--model        Path to the trained pth model.
--config       The config file used for training the model.
```
#### Config file details️

Bellow we detail the model's parameters that can be controlled in the config file `configs/config.json`.

```javascript
{
    "name": "DnCNN",                         // name
    "experim_name": "DnCNN_dummy",           // name of the experiment 
    "seeds": [1,2],                          // seeds used to initialize the weight's model
    "sigma": 0.1,                            // variance of the gaussian noise added

    "model":{
        "depth": 7,                          // number of layer of the model minus 2
        "n_channels": 64,                    // number of activation map 
        "image_channels": 1,                 // number of channel of the image (grayscale 1, RGB 3, ...)
        "kernel_size": 3,
        "padding": 1,
        "architecture": "direct",            // architecture of the network either direct or residual
        "spectral_norm": true                // boolean flag to use spectral normalization
    },


    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 1e-3,
            "weight_decay": 0
        }
    },


    "train_loader": {
        "data_dir": "data/train_dummy",
        "batch_size": 10,
        "shuffle": true,
        "num_workers": 0
    },


    "val_loader": {
        "data_dir": "data/val_dummy",
        "batch_size": 1,
        "shuffle": false,
        "num_workers": 0
    },

    "test_loader": {
        "data_dir": "data/test_dummy",
        "batch_size": 1,
        "shuffle": false,
        "num_workers": 0
    },

    "trainer": {
        "epochs": 20,
        "save_dir": "saved/",
        "save_period": 5,
        
        "tensorboardX": true,
        "log_dir": "saved/",
        "log_per_iter": 20,

        "val": true,
        "val_per_epochs": 5
    }
},
```