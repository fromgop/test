from flask import Flask, render_template, request, redirect
import os
import cPickle
import logging
import caffe
import urllib
import werkzeug
import exifutil
import datetime

app = Flask(__name__)

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOADS_FOLDER = '/tmp/caffe_demo_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

@app.route('/')
def mainPage():
    return render_template('MainPage.htm', has_result=False)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    image_url = request.args.get('image_url', '')
    try:
        str_buffer = StringIO.StringIO(urllib.urlopen(image_url).read())
        image = caffe.io.load_image(str_buffer)
    except Exception as err:
        logging.info('URL Image open error: %s' %err)
        return render_template('MainPage.htm', has_result=True,result=(False, 'Cannot open image from URL.'))
        logging.info('Image: %s', image_url)
        result = app.clf.classify_image(image)
        return render_template('index.html', has_result=True, result=result, imagesrc=image_url)

@app.route('/classify_upload', methods=['POST'])
def classifyy_upload():
    try:
        imagefile = request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOADS_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
        
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return render_template('index.html', has_result=True,result=(False, 'Cannot open uploaded image.'))
    
    logging.info('Image: %s', image_url)
    result = app.clf.classify_image(image)
    
    