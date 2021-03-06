FROM pytorch/pytorch
WORKDIR '/app'

USER root

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libgtk2.0-dev -y
RUN pip install opencv-python mypy

COPY ./src .

CMD ["mypy", "train.py"]