This is the project page associating to our work on Face Sketch Synthesis:

Zhang, S., Ji, R., Hu, J., Lu, X., Li, X., "<a href=https://ieeexplore.ieee.org/abstract/document/8478205>Face Sketch Synthesis by Multidomain Adversarial Learning.</a>" TNNLS, 2018.

This page contains the codes for our model "MDAL". If you have any problem, please feel free to contact us.

# Prerequisites

* Python (2.7 or later)
* numpy
* scipy
* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1
* pyTorch 0.3

# Training & Test

After preparing the training/test images, run:
```
./run.sh
```
The example of runing the training phase:
```
python train.py --dataset XM2VTS --nEpochs 200 --cuda
```
The example of runing the test phase:
```
python test.py --dataset XM2VTS --p_model photo_G_1_model_epoch_200.pth --s_model sketch_G_2_model_epoch_200.pth --cuda
```

# Generated Sketches

The generated sketches are avaliable now at:

## Baidu Netdisk

Link: https://pan.baidu.com/s/1LCSZRKDQskfsycDH3TNopA
Password: j4wu


or

## Google Drive:

Link: https://drive.google.com/file/d/14468pBbin34TtbvxsFOYYKBkQdoVTm80/view


# Demo

The demo of this paper is avaliable at:

Link: http://mac.xmu.edu.cn/demos.html

If you want to deploy the server by yourslef, you can checkout demo branch.

## Prerequisites
python3
flask

## Run web server
cd $MDAL_ROOT/flask
python3 app.py
URL: http://localhost:5002 

You may need to configure chrome to read camera devices under the HTTP protocol. You can modify params such as $G_USE_CUDA in $MDAL_ROOT/flask/sketch.py, and we recommend using GPU when runnig the server.

