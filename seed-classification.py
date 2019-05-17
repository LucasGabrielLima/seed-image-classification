import cv2, os
import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

#classes
#----------------------------------------------------
class DB:
	data = []

	def __init__(self, root_folder):
		#lists directories
		directory_list = list()
		for root, dirs, files in os.walk(root_folder, topdown=False):
			for name in dirs:
				directory_list.append((name, os.path.join(root, name)))
		#gets images from every listed directory
		images = list()
		for name, directory in directory_list:
			images = self.load_images_from_folder(directory, name)
			#appends images from folder to data
			if len(images) > 0:
				self.data = self.data + images	

	def load_images_from_folder(self, folder, classification):
		images = list()
		files = os.listdir(folder)
		for filename in files:
			if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.JPG']]):
				img = cv2.imread(os.path.join(folder, filename))
				if img is not None:
					image = Image(filename, classification, img)
					images.append(image)
		return images

	def getData(self):
		return self.data

class Image:
	image = ''
	name = ''
	classification = ''
	binarized = ''
	features = ''

	def __init__(self, name, classification, image):
		self.image = image
		self.name = name
		self.classification = classification

#methods
#----------------------------------------------------
def show(image):
	if(DEBUG):
		cv2.imshow("img", image)
		cv2.waitKey(10000000)

# def bgRemoval(images):
	# lower = np.array([126,126,126])  #-- Lower range --
	# upper = np.array([127,127,127])  #-- Upper range --
	# mask = cv2.inRange(img, lower, upper)
	# res = cv2.bitwise_and(img, img, mask= mask)  #-- Contains pixels having the gray color--
	# cv2.imshow('Result',res)

def binarization(images):
	for img in images:
		img.binarized = threshold(img.image)

def threshold(image):
	new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	show(new_image)
	ret,binary = cv2.threshold(new_image,105,255,cv2.THRESH_BINARY_INV)
	show(binary)

#====================================================
#main
#====================================================

#Loads Database
db = DB('./images')
images = db.getData()

#Segmentation
# bgRemoval(images)

#binarization(images)

#Printing test
# for img in images:
	# print(img.name, img.classification)