FROM pytorch/pytorch
WORKDIR '/app'
COPY ./src .

USER root

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y & apt-get install libgtk2.0-dev -y
RUN pip install opencv-python
RUN pip install mypy
RUN chmod +x ./checking.sh

CMD ["./checking.sh"]