from PIL import Image # Notice the 'from PIL' at the start of the line
import os

im = Image.new("RGB", (200, 30), "#ddd")
im.save(os.path.join("/home/muktevigk", "x.png") )
#print (os.path.realpath(im))