import os
import sys
import caffe
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
classList = []

for line in open('/home/muktevigk/deep-learning/caffe/data/ilsvrc12/synset_words.txt','r'):
   classList.append(" ".join(line.split()[1:]))

#print classList


MODEL_FILE = '/home/muktevigk/deep-learning/caffe/models/bvlc_reference_caffenet/deploy.prototxt'

PRETRAINED = '/home/muktevigk/deep-learning/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

MEAN_FILE = '/home/muktevigk/deep-learning/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

IMAGE_FILE = '/home/muktevigk/deep-learning/caffe/examples/images/aa.jpg'

LABEL_FILE = '/home/muktevigk/deep-learning/caffe/data/ilsvrc12/synset.txt'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(MEAN_FILE).mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255, image_dims=(256, 256))

input_image = caffe.io.load_image(IMAGE_FILE)

plt.imshow(input_image)
plt.savefig('/home/muktevigk/deep-learning/caffe/outputs/24.png')
plt.close()

outputFile=[]
opf=open("output.txt","w")

prediction = net.predict([input_image])
print 'predicted class:', prediction[0].argmax()

count=0
for scores in prediction[count]:
	count=count+1
	if scores > 0.01:
		opf.write("%s, %s\n" %(count,scores))
		print classList[count-1]
		
opf.close()
for line in open("output.txt","r"):
			prob=line.split(",")[0]
			classno=line.split(",")[1]
			outputFile.append((prob,classno[:-1]))



#print outputFile
print 'predicted class:', prediction[0].argmax()



plt.plot(prediction[0])
plt.savefig('/home/muktevigk/deep-learning/caffe/outputs/prediction.png')



