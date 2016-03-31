from flask import Flask, render_template, request
import os
import time
#import cPickle
import numpy as np
#import pandas as pd
from PIL import Image
#import logging
import caffe
import urllib
#import matplotlib.pyplot as plt
import werkzeug
import exifutil
#mport datetime
import cStringIO as StringIO

app = Flask(__name__)

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '../../../')
UPLOADS_FOLDER = '/tmp/caffe_demo_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
MODEL_FILE = '/home/muktevigk/deep-learning/caffe/models/SUN397/deploy.prototxt'
PRETRAINED = '/home/muktevigk/deep-learning/caffe/models/SUN397/sun397_iter_110000.caffemodel'
MEAN_FILE = '/home/muktevigk/deep-learning/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
LABEL_FILE = '/home/muktevigk/deep-learning/caffe/data/SUN397/classes_des.txt'
classlist=[]
for line in open('/home/muktevigk/deep-learning/caffe/data/SUN397/classes_des.txt', 'r').readlines():
	d = '\n'.join(line.split("\n"[:1]))
	classlist.append("%s" %d)
metalist =[] 
for line in open('/home/muktevigk/deep-learning/caffe/data/SUN397/class_desc_scores.txt', 'r').readlines():
    desc = ''.join(line.split("\n"[:1]))
    metalist.append(desc)

image_dim = 256
@app.route('/')
def mainPage():
    return render_template('index.html', has_result=False)
    
@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = request.args.get('imageurl', '')
    print imageurl      
    string_buffer = StringIO.StringIO(
    urllib.urlopen(imageurl).read())
    image = caffe.io.load_image(string_buffer)    
    result = classify_image(image)
    return render_template('index.html', has_result=True, result=result, imagesrc=imageurl)
	
@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    imagefile = request.files['imagefile']
    filename_ = werkzeug.secure_filename(imagefile.filename)
    filename = os.path.join(UPLOADS_FOLDER,filename_)
    imagefile.save(filename)
    image = exifutil.open_oriented_im(filename)
    result = classify_image(filename)
    print result
    imagesrc=embed_image_html(image)
    return render_template('index.html', has_result=True, result=result, imagesrc=imagesrc)
        
def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS)
    
def classify_image(image):
    starttime = time.time()    
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(MEAN_FILE).mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255)
    input_image = caffe.io.load_image(image)
    endtime = time.time()
    Time = endtime-starttime
    print image
    meta=[]
    topScores=[]
    prediction = net.predict([input_image])
    topScores = []
    count = 0
    for scores in prediction[count]:        
        count = count+1
        if scores > 0.1:
            topScores.append((scores, count))
            #print (classlist[count-1])
            sortedScoresList = sorted(topScores, reverse=True)
    #print ()
    if cmp(classlist, metalist):
        print "yes"
    else:
        print "No"
    for scores, count in sortedScoresList:
        l = ''.join(classlist[count-1])
        meta.append([l, str(scores)])
        #meta1 = ''.join(meta)        
        print l
    print meta
    return (meta, '%.3f' %Time)
        
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=int(5001), debug=True)