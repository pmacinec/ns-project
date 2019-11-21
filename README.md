# Fake news detection

**Authors:** Peter Mačinec & Simona Miková


## Installation and running

To run this project, please make sure you have Docker installed and follow the steps (also make sure to have [nvidia docker](https://github.com/NVIDIA/nvidia-docker) installed if you want to train on your Nvidia graphics card):
1. Clone this repository with command:
    ```shell script
    $ git clone git@github.com:pmacinec/ns-project.git
    ```
1. Get into directory with command `cd ns-project/`.
1. Build image using command:
    ```shell script
    $ docker build -t ns_project docker/
    ```
1. Run docker container using command:
    ```shell script 
    $ docker run --gpus all -it --name fake_news_detection_con --rm -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/project/ ns_project
    ```
   **Note:** If you don't have Nvidia graphics card with nvidia docker installed, skip `--gpus all` argument in script call.


## Training proposed neural network

Training proposed neural network has to be done in running docker container `fake_news_detection_con` (please, see section **Installation and running**).
1. Get into docker container with command:
    ```shell script
    $ docker exec -it fake_news_detection_con bash
    ```
1. Run training script with arguments you need (arguments are used to configure neural network model and training process):
    ```shell script
    $ python src/model/train.py {args}
    ```
    **Note:** All arguments are listed in section **Training configuration**.


## Data retrieval

For training our neural network, we used data from Monant platform. Data can be retrieved by following steps:
1. Add config due to your credentials into `src/data/retrieval/config.json` to Monant platform:
    ```json
    {
        "username": "YOUR-USERNAME",
        "password": "YOUR-PASSWORD",
        "api_host": "MONANT-API-HOST",
        "data_folder": "data/raw"
    }
    ```
   If you don't have your own credentials, check [Monant platform documentation](https://documenter.getpostman.com/view/8615295/SVtPWq1j?version=latest) to next steps. 
1. In repository root, run command:
    ```shell script
    $ python src/data/retrieval/data_saver.py
    ```
1. According to above config, data will be stored in `data/raw` folder.


## Training configuration
TODO