# IRN
<div align=center>

[![Build Status](https://travis-ci.com/TheStarkor/IRN.svg?branch=main)](https://travis-ci.com/TheStarkor/IRN)

</div>

implementation of paper: Invertible Image Rescaling (ECCV 2020 Oral)

![model](https://github.com/pkuxmq/Invertible-Image-Rescaling/raw/master/figures/architecture.jpg)

## 유의 사항 
연구실 과제에 사용한 코드가 있어, 외부 배포를 금지합니다. 

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
