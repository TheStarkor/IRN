# IRN
<div align=center>

[![Build Status](https://travis-ci.com/TheStarkor/IRN.svg?branch=main)](https://travis-ci.com/TheStarkor/IRN)

</div>

implementation of paper: Invertible Image Rescaling (ECCV 2020 Oral)

![model](https://github.com/pkuxmq/Invertible-Image-Rescaling/raw/master/figures/architecture.jpg)

## Get Started

### Docker
```
$ docker build -t irn -f Dockerfile.train
$ docker run --gpus all irn
```

### Local
```
$ ./src/run.sh
```
