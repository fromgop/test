import os
import sys
import caffe
import numpy as np
import matplotlib
import time as t
import matplotlib.pyplot as plt

topScores =[]

MODEL_FILE = '/home/placesCNN/placesCNN/models/bvlc_reference_caffenet/deploy.prototxt'

PRETRAINED = '/home/placesCNN/placesCNN/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

MEAN_FILE = '/home/placesCNN/placesCNN/python/caffe/imagenet/ilsvrc_2012_mean.npy'
LABEL_FILE = '/home/placesCNN/placesCNN/data/ilsvrc12/synset.txt'
net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(MEAN_FILE).mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255, image_dims=(256, 256))

#st = t.time()
classList = []

for line in open('/home/placesCNN/placesCNN/data/ilsvrc12/synset_words.txt','r'):
   classList.append(" ".join(line.split()[1:]))

#print classList
img_list_file=open("filelist.txt","r")

for imagefile in img_list_file.readlines():
    
# IMAGE_FILE = '/home/placesCNN/placesCNN/examples/images/15.jpg'
    IMAGE_FILE=imagefile[:-1]

    input_image = caffe.io.load_image(IMAGE_FILE)
    plt.imshow(input_image)
    #plt.savefig('/home/placesCNN/placesCNN/outputs/15.png')
    plt.close()
    topScores=[]
    ofilename=".."+''.join(IMAGE_FILE.split(".")[:-1]) + ".lab"
    print ofilename
    opf1=open(ofilename,"w")
    prediction = net.predict([input_image])
    print 'predicted class:', prediction[0].argmax()
    outputList=[]
    opf=open('output.txt','w')
    count=0

#TODO: SORT THE SCORES
    for scores in prediction[count]:
        count=count+1
        if scores > 0.01:
            topScores.append((scores,count))
    opf.close()
    sortedScoresList=sorted(topScores, reverse=True)
    for scores,count in sortedScoresList:
        opf1.write("%s | %s\n" %(classList[count-1], scores))
#sortedlist=sorted(topScores, reverse=True)
#print sortedlist



#print outputFile
print 'predicted class:', prediction[0].argmax()



#plt.plot(prediction[0])
#plt.savefig('/home/placesCNN/placesCNN/outputs/prediction.png')

#et2 = t.time() - (st + et1)

#print et2

