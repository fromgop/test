import os
import sys
import caffe
import Image
import numpy as np
import matplotlib
#import cv2
import matplotlib.pyplot as plt

classlist=[]
for line in open('/home/muktevigk/deep-learning/caffe/data/SUN397/classes_des.txt', 'r').readlines():
	d = (" ".join(line.split(","[:1])))
	classlist.append("%s" %d)
	

MODEL_FILE = '/home/muktevigk/deep-learning/caffe/models/SUN397/deploy.prototxt'

PRETRAINED = '/home/muktevigk/deep-learning/caffe/models/SUN397/sun397_iter_110000.caffemodel'

MEAN_FILE = '/home/muktevigk/deep-learning/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

IMAGE_FILE = '/home/muktevigk/deep-learning/caffe/examples/images/sun_drehneatnzkcoghw.jpg'

LABEL_FILE = '/home/muktevigk/deep-learning/caffe/data/SUN397/classes_des.txt'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(MEAN_FILE).mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255)

input_image = caffe.io.load_image(IMAGE_FILE)
print input_image.shape
resizedImage = caffe.io.resize_image(input_image,[256,256])
print resizedImage.shape

plt.imshow(input_image)
plt.savefig('/home/muktevigk/deep-learning/caffe/outputs/house.jpg')
plt.close()

prediction = net.predict([input_image])
print 'predicted class:', prediction[0].argmax()
topScores=[]

count=0
for scores in prediction[count]:
	count=count+1
	if scores > 0.01:
		topScores.append((scores, count))
		#print topScores
		print classlist[count-1]
		sortedScoresList=sorted(topScores, reverse=True)
		#print (sortedScoresList)
for scores, count in sortedScoresList:
	print ("%s  %s\n" %(classlist[count-1], scores))
print 'predicted class:', prediction[0].argmax()
plt.plot(prediction[0])
plt.savefig('/home/muktevigk/deep-learning/caffe/outputs/prediction.png')

