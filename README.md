# Fake news detection

**Authors:** Peter Mačinec & Simona Miková


## Installation and running

To run this project, please make sure you have Docker installed and follow the steps:
1. Clone this repository with command `git clone git@github.com:pmacinec/ns-project.git`.
1. Get into directory with command `cd ns-project/`.
1. Build image using command `docker build -t ns_project docker/`.
1. Run docker container using command `docker run -it --rm -u $(id -u):$(id -g) -p 8888:8888 -v $(pwd):/project/ ns_project`.
