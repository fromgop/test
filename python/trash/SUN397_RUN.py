import os
import sys
import caffe
import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
classList=[]
for line in open('/home/muktevigk/deep-learning/caffe/data/SUN397/classes_des.txt','r'):
	d = (" ".join(line.split(","[:1])))
	classList.append("%s" %d)

MODEL_FILE = '/home/muktevigk/deep-learning/caffe/models/SUN397/deploy.prototxt'

PRETRAINED = '/home/muktevigk/deep-learning/caffe/models/SUN397/caffenet_train_iter_450000.caffemodel'

MEAN_FILE = '/home/muktevigk/deep-learning/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

IMAGE_FILE = '/home/muktevigk/deep-learning/caffe/examples/images/sun_ayfvjufwlxjspvee.jpg'

LABEL_FILE = '/home/muktevigk/deep-learning/caffe/data/SUN397/classes_des.txt'

def resize(input_image, width=256, height=256):
	im1 = Image.open( ImageName)
	#input_image = caffe.io.load_image(IMAGE_FILE)
	im2 = im1.resize((width, height), Image.NEAREST)
	#resizedImage = caffe.io.resize_image(input_image,[256,256])
	#im2.save(os.path.join(app.config['RESIZED_IMAGE_FOLDER'], imageName))
	return im2

net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(MEAN_FILE).mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255, image_dims=(256, 256))

#resizedImage = resize(IMAGE_FILE)

input_image = caffe.io.load_image(IMAGE_FILE)
resizedImage = caffe.io.resize_image(input_image,[256,256])
plt.imshow(resizedImage)
plt.savefig('/home/muktevigk/deep-learning/caffe/outputs/15.png')
plt.close()

ofilename=IMAGE_FILE.split()

prediction = net.predict([resizedImage])
print 'predicted class:', prediction[0].argmax()


count=0
for scores in prediction[count]:
	count=count+1
	if scores > 0.02:
		#opf.write("%s, %s\n" %(scores,count))
		#print ("%s, %s" %(scores, count))
		print classList[count-1]

print 'predicted class:', prediction[0].argmax()



plt.plot(prediction[0])
plt.savefig('/home/muktevigk/deep-learning/caffe/outputs/prediction.png')



