# Logo Classification using 3D and 2D Augmentation techniques

## Setup

### Requirements
1. Python 3.6 <
2. requirements.txt
3. docker

## Usage

First the 3D generator needs to be set up. We created a docker container that lets you run it in different environments. All the needed files are in the generator directory.
1. unzip the compressed file and replace the logos directory with the dataset you want to use for training.
2. run `buildImage.sh`
3. Modify the path to your generator in `startContainer.sh` then run it.

Now you should have an instance of the generator ready to use.
If you want to check if it is running correctly you can run `execBashOnContainer.sh` and then run `open_Player.log.sh` - If it says "Waiting for Connection..." it is correctly running.

