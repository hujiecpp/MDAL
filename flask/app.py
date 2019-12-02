import os
import sys
sys.path.append('../')

from flask import Flask, render_template, Response
from flask import request, json, jsonify
from utils import loadImage
from torch.autograd import Variable
import base64


from time import time, strftime
import cv2
from scipy.misc import imresize, imsave
from sketch import Sketch, TimePoint
import numpy as np
from cam_local import Camera


app = Flask(__name__) 
sketch = Sketch()
tp = TimePoint()

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/mobile') 
def index_mobile():
    return render_template('image_mobile.html')

@app.route('/auto') 
def index_auto():
    return render_template('webcam_auto.html')

@app.route('/click') 
def index_click():
    return render_template('webcam_click.html')

@app.route('/upload') 
def index_upload():
    return render_template('image_upload.html')

@app.route('/server') 
def index_local():
    return render_template('webcam_server.html')


@app.route('/cam_remote', methods=['POST'])
def cam_remote():
    tp.start()
    img_str = request.get_data() # <class 'bytes'>; b'data:image/jpeg;base64,...'
    img_str = img_str.split(b',')[1]
    # print(type(img_str),img_str)
    img_str = base64.b64decode(img_str) # <class 'bytes'>
    img_np = np.frombuffer(img_str, np.uint8) # <class 'numpy.ndarray'> (34232,)
    img_np = cv2.imdecode(img_np, 1) # <class 'numpy.ndarray'> (480, 640, 3); flag=1, 8 bits，3 channels
    # cv2.imwrite('static/images/base64.jpg',img_np)
    if len(img_np.shape) is not 3:
        res_str = "request image error; img type:{}; img shape:{}".format(type(img_np), img_np.shape)
        print(res_str)
        return res_str

    img_in = sketch.preProcess(img_np)
    tp.getCost('pre')
    wp = sketch.photo_G_1(img_in)
    out = sketch.sketch_G_2(wp)
    tp.getCost('net')
    img_out = sketch.postProcess(out) # <class 'numpy.ndarray'> (480, 640, 3)
    img_resp = cv2.imencode('.jpg', img_out)[1].tostring() # <class 'bytes'>
    img_resp = base64.b64encode(img_resp) # <class 'bytes'>
    img_resp = "data:image/jpeg;base64,{}".format(img_resp.decode()) # <class 'str'>
    tp.getCost('post')
    tp.getTotalCost()

    return img_resp


#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])
# save the image as a picture
@app.route('/img_upload', methods=['POST'])
def img_upload():
    tp.start()
    req_file = request.files['file']  # get the image
    print(type(req_file),req_file) # <class 'werkzeug.datastructures.FileStorage'> <FileStorage: 'blob' ('image/jpeg')>
    name_sub = req_file.filename.split('.')[1].lower()
    if name_sub not in ALLOWED_EXTENSIONS:
        res_str = "request image error: format not support"
        print(res_str)
        return res_str 

    img_name = 'static/images/upload.{}'.format(name_sub)
    req_file.save(img_name)

    img = loadImage(img_name, -1, -1, -1, 'Testing')
    img_in = Variable(img).view(1, -1, sketch.IMG_SIZE_IN, sketch.IMG_SIZE_IN)
    if sketch.use_cuda:
        img_in = img_in.cuda()
    tp.getCost('pre')
    wp = sketch.photo_G_1(img_in)
    out = sketch.sketch_G_2(wp)
    tp.getCost('net')
    img_out = sketch.postProcess(out) # <class 'numpy.ndarray'> (480, 640, 3)
    img_resp = cv2.imencode('.jpg', img_out)[1].tostring() # <class 'bytes'>
    img_resp = base64.b64encode(img_resp) # <class 'bytes'>
    img_resp = "data:image/jpeg;base64,{}".format(img_resp.decode()) # <class 'str'>
    tp.getCost('post')
    tp.getTotalCost()

    return img_resp


@app.route('/cam_server')
def cam_server():
    return Response(gen(Camera()), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def gen(cam):
    """Video streaming generator function."""
    if not sketch.need_trans: # mv if to outside
        while True:
            img_in = cam.get_frame()
            # encode as a jpeg image and return it
            img_out = cv2.imencode('.jpg', img_in)[1].tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img_out + b'\r\n')
    else:
        while True:
            # tp.start()
            img = cam.get_frame()
            img_in = sketch.preProcess(img)
            # tp.getCost('pre')
            wp = sketch.photo_G_1(img_in)
            out = sketch.sketch_G_2(wp)
            # tp.getCost('net')
            img_out = sketch.postProcess(out)
            # img = cv2.resize(img, (sketch.IMG_W, sketch.IMG_H)) # size between img_cam and img_out may diff
            img_out = np.hstack((img, img_out)) 
            # print(img_out.shape, img_out.dtype)
            # cv2.imwrite('static/images/cam_test.jpg',img_out)
            img_out = cv2.imencode('.jpg', img_out)[1].tobytes()
            # tp.getCost('post')
            # tp.getTotalCost()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img_out + b'\r\n\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002 , threaded=True)