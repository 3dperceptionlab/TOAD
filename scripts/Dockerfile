FROM huggingface/transformers-pytorch-gpu:latest

RUN apt-get update && apt-get install -y git gcc libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
