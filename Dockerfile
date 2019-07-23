FROM tensorflow/tensorflow:1.13.0rc1-gpu-py3


ENV TF_ROOT=/opt/TF_ROOT/

RUN apt-get update && apt-get install -y --no-install-recommends \
python3-tk \
python3-pydot \
python3-pip

COPY requirements.txt TF_ROOT/requirements.txt

WORKDIR TF_ROOT

RUN pip3 install -r requirements.txt

#RUN pip3 install --upgrade keras opencv-python scipy numpy sklearn hyperas decorator==4.3.0 pillow matplotlib
