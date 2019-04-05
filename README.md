# Manual to 3D Classification Survey

## Abstract

We compiled set of publicly available neural networks for classification of 3D models. The code works with ModelNet40 and ShapeNetCore datasets which are also available online. This is a manual explaining how to convert datasets, train and test these networks.

## Requirements

To run the code you will need a computer with Linux operating system and NVIDIA GPU.

You will need to install:

*   NVIDIA drivers ([Installation guide here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation))
*   Docker version 1.12 or higher ([Installation guide here](https://docs.docker.com/install/))
*   NVIDIA Container Runtime for Docker ([Installation guide here](https://github.com/NVIDIA/nvidia-docker))

And that is all! Each neural network is an independent Docker image and all its dependencies are installed when building the image. All code is written in Python.

## Datasets Setup

The code is made to work with ModelNet40 and ShapeNetCore datasets. The easiest way to run it with custom dataset is to restructure your data so it copies the structure of one of these datasets.

*   ### ModelNet40

    *   Get the dataset [here](http://modelnet.cs.princeton.edu/). For experiments we used manually aligned version which can be downloaded [here](https://github.com/lmb-freiburg/orion).

	*   Unpack the downloaded archive and you are ready to go!

*  ###  ShapeNetCore

	*   Get the dataset [here](https://www.shapenet.org/download/shapenetcore). You need to register and wait for confirmation email.
	*   Unpack the downloaded archive.

## General Setup

You can download all the code from Github [here](https://github.com/Alobal123/diplomka).

Each network is implemented as a separate Docker image. To learn more about Docker, images and containers visit [this page](https://docs.docker.com/get-started/).

Each neural network is contained in one directory in /dockers. None of the networks accepts mesh files as their input directly, so some data conversion is required. All data conversion is implemented in docker with same structure as neural networks themselves. Code for data conversion is located in /dockers/data_conversion.

Each directory contains two important files - config.ini and run.sh, which you will need to open and edit. Another important file is Dockerfile which contains the definition of the docker image. Remaining files contain the code which differ from the original network implementation. Original network code is downloaded automatically when building the image and you can find the download link below.

run.sh is a runnable script which builds the docker image, runs the docker container and executes the neural network or data conversion. You will need to setup a couple of variables here:

*   name - will be used as a name of the docker image and docker container. You can leave this at default value unless it is in conflict with some already existing image or you want to run more instances of this image at once. With data conversion scripts the name is the name of the converted dataset and directory of the same name will be created. The name of the image can be changed by changing variable image_name in this case.
*   dataset_path - contains path to the root directory of the dataset on your filesystem.
*   out_path - contains path to the directory where training logs and network weights will be saved. This directory will be created automatically.
*   GPU - index of GPU which will be visible to docker container. Has to be a single integer. We currently do not support multiple GPUs.
*   docker_hidden - Must be one of t or d. With t the container will be run in interactive mode, meaning it will run in your console. With d it will in detached mode i.e. in the background. For more information check docker documentation [here](https://docs.docker.com/engine/reference/run/).

config.ini contains most of the relevant parameters of the network or data conversion. The file is split to sections where each section is started by [SECTION] statement. Then on each line a parameter in format key = value. You can find explanation of network parameters in later sections.

## Data conversion

To convert your dataset you need to set the parameters described above and then simply run script run.sh in your console. This will convert the dataset to various formats directly readable by the neural networks.

Parameters for data conversion in config.ini file:

*   data - path to the dataset inside the container. Does not have to be changed.
*   output - path to the directory inside the container where converted dataset will be saved. Does not have to be changed.
*   log_file - path and name of the file where progress of the data conversion will be written.
*   num_threads - maximum number of threads to use.
*   dataset_type -  which dataset is converting. Must be one of modelnet or shapenet currently.

For more detail about individual data conversion scripts, continue [here](conversions.md).

## Neural Networks

Each of the neural networks is implemented in Python but in different framework. That is why we used the Docker infrastructure. We try to present a unified framework to easily test and train the networks without changing the code. This section will briefly introduce the used networks and some of their most important parameters.

Parameters common to all neural networks:

*   name - will be used as the name of the experiment used in log files.
*   data - path to the dataset inside the container. Does not have to be changed.
*   log_dir - path to the directory inside the container where logs and weights will be saved. Does not have to be changed.
*   num_classes - number of classes in the dataset.
*   batch_size - size of the batch for training and testing neural networks.
*   weights - if you want to test or finetune already trained network, this should be the number of this model. If you want to train from scratch, this should be -1\.
*   snapshot_prefix - name of the file where weights will be saved. Number of training epoch when these weights are saved will be added to this.

*   max_epoch - number of epochs to train for. One epoch means one pass through the training part of the dataset.
*   save_period - the trained network will be saved every epoch divisible by save_period.
*   train_log_frq - frequency of logging during training. It is roughly number of examples seen by network.

*   test - if you want to only test already trained network, set this to True. weights parameter has to have a valid value bigger than -1\. Should be False for training.

For more details about individual networks, continue [here](networks.md).

## Logging and Evaluation

Our framework offers some basic logging options. It saves several .csv to the logging directory. The logger keeps track of time of the training, training epochs and some other value. By default four values are tracked: training loss, training accuracy, test loss and test accuracy. Evaluation on the test set is performed after each epoch of training. Also some basic graphs using matplotlib library are created and saved during training. When testing your already trained network (using test = True) text file is saved where network category prediction is saved along with a simple html confusion matrix.
