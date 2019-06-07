import cv2, os
import numpy as np
# import matplotlib.pyplot as plt
import math
DEBUG = False

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
		for filename in files[0:10]:
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

def binarization(images):
	for img in images:
		img.image = threshold(img.image)

def threshold(image):
	(b, g, r ) = image[0][0]
	show(image)
	image = cv2.medianBlur(image, 3)
	new_image = np.empty((image.shape[0], image.shape[1]), dtype=np.uint8)
	print(image.shape[0], image.shape[1])
	for i in range(0, image.shape[0]):
		for j in range(0, image.shape[1]):
			new_image[i][j] = 0 if compareColors((b, g, r), image[i][j]) else 255
	return new_image

def compareColors(bg, color):
	for i in (0, 2):
		a = int(bg[i])
		b = int(color[i])
		if np.abs(b-a) > 48:
			return False
	return True

def segmentation(images):
	for img in images:
		aux = img.image.copy()

		# Apenas para testes
		countourImg = aux.copy()
		contours, hier = cv2.findContours(img.image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		# for cnt in contours:
		# 	# print('countour')
		# 	cv2.drawContours(img.image,[cnt],0,(255,255, 255),3)

		centres = []

		for cnt in contours:
			if cv2.contourArea(cnt) < 1000:
		  		continue
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(img.image,[box],0,(255,255,255),5)


		print img.name
		cv2.imshow('teste', img.image)
		cv2.waitKey(0)









#====================================================
#main
#====================================================

#Loads Database
db = DB('./images')
images = db.getData()
print("Finished Loading Database")

#Segmentation
binarization(images[:5])
print("Finished Binarization")

segmentation(images[:5])

#Printing test
# for img in images:
	# print(img.name, img.classification)
