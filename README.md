# dartspose

Dartspose is a system that uses openpose to measure the angle of the person throwing the darts,
and then measures the similarity of the data to that of a good person.

This system was created as a demo.

This repository is  customized for the `M1mac`.

## My method

### Measure the angle of the person throwing the darts

I use tf-pose-estimation of gsethi2409.

Link to gsethi2409's repo: https://github.com/gsethi2409/tf-pose-estimation

### Measure of similarity to someone who is good at it

Using Spring what is partial match DTW.

## Download

Running in the terminal

```bash
https://github.com/yu-sakana/dartspose.git
```

## Requirements

* python 3.8
* tf_slim


```bash
conda env create -n {your env name} -f darts_pose.yml
```

## Initial Setup

First,create a video directory in dartspose

```bash
mkdir video
```

Second, edit path on line 193 of the darts_convert.py.

```
/Users/saka/Documents/seminar/pose/tf-pose-estimation/
↓
/your path/dartspose
```

#Quick start
Dartspose works with the following code.

```bash
sh dartspose.sh
```

