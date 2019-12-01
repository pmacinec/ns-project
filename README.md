# Fake news detection

**Authors:** Peter Mačinec & Simona Miková


## Installation and running

To run this project, please make sure you have Docker installed and follow the steps (also make sure to have [nvidia docker](https://github.com/NVIDIA/nvidia-docker) installed if you want to train on your Nvidia graphics card):
1. Clone this repository with command:
    ```shell script
    git clone git@github.com:pmacinec/ns-project.git
    ```
1. Get into directory with command `cd ns-project/`.
1. Build image using command:
    ```shell script
    docker build -t ns_project docker/
    ```
1. Run docker container using command:
    ```shell script 
    docker run --gpus all -it --name fake_news_detection_con --rm -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/project/ ns_project
    ```
   **Note:** If you don't have Nvidia graphics card with nvidia docker installed, skip `--gpus all` argument in script call.


## Training proposed neural network

Training proposed neural network has to be done in running docker container `fake_news_detection_con` (please, see section **Installation and running**).
1. Get into docker container with command:
    ```shell script
    docker exec -it fake_news_detection_con bash
    ```
1. Run training script with arguments you need (arguments are used to configure neural network model and training process):
    ```shell script
    python src/model/train.py {args}
    ```
    **Note:** All arguments are listed in section **Training configuration**.

All training logs are stored in `logs` folder and model checkpoints in `models` folder. By default, concrete training logs and checkopint models are stored in timestamp folder inside `logs` or `models`  folders. For using custom folder name instead of timestamp, use script call argument `--name` when starting training.


## Training configuration

To configure training and neural network model, two options are available:
1. Use arguments when calling training script:

    |       Argument       | Short   | Value type | Description |
    |----------------------|:-------:|:----------:|-------------|
    | `--file`             | `-f`    | `<str>`    | path to custom config file (discussed in second part) |
    | `--batch-size`       | `-bs`   | `<int>`    | batch size to be used in training |
    | `--learning-rate`    | `-lr`   | `<float>`  | learning rate to be used in training |
    | `--num-hidden-layers`| `-hl`   | `<int>`    | number of hidden layers |
    | `--epochs`           | `-e`    | `<int>`    | number of epochs to train |
    | `--max-words`        | `-w`    | `<int>`    | maximum words in vocabulary to use |
    | `--samples`          | `-s`    | `<int>`    | number of samples from data (by default, all are used) |
    | `--data`             | `-d`    | `<str>`    | not required - path to data csv file |
    | `--test-size`        | `-t`    | `<float>`  | train test split rate (test size) |
    | `--max-sequence-len` | `-sl`   | `<int>`    | maximum length of all sequences |
    | `--lstm-units`       | `-lstm` | `<int>`    | number of units in LSTM layer |
    | `--name`             | `-n`    | `<str>`    | training name - also folder name for logs and checkpoint model |
1. Write custom config file (in JSON format) and pass path to it as train script call argument (`--file`/`-f`). **Remember, that script call arguments replace config file arguments!** Example of config file:
    ```json
    {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 15,
        "lstm_units": 64
    }
    ```
   Allowed parameters: `batch_size`, `learning_rate`, `num_hidden_layers`, `epochs`, `max_words`, `num_samples`, `data_file`, `test_size`, `max_seq_len`, `lstm_units`. For description of those values, see table above (all values can be semantically mapped to arguments in table).



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
    python src/data/retrieval/data_saver.py
    ```
1. According to above config, data will be stored in `data/raw` folder.
