# Fake news detection

**Authors:** Peter Mačinec & Simona Miková


## Installation and running

To run this project, please make sure you have Docker installed and follow the steps:
1. Clone this repository with command `git clone git@github.com:pmacinec/ns-project.git`.
1. Get into directory with command `cd ns-project/`.
1. Build image using command `docker build -t ns_project docker/`.
1. Run docker container using command `docker run --gpus all -it --rm -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/project/ ns_project`.


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
1. In repository root, run command `python src/data/retrieval/data_saver.py`.
1. According to above config, data will be stored in `data/raw` folder.
