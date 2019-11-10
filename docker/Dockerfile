FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter

RUN mkdir project

WORKDIR /project/

COPY ./requirements.txt /project/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir ~/nltk_data && python -c "import nltk; nltk.download(['punkt', 'stopwords'], download_dir='/usr/share/nltk_data')"

ENTRYPOINT jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
