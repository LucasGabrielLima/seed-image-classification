import cv2, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skm
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import axes3d
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
	pb = ''
	features = ''
	objects = []
	classificationObjects = []
	objects_bin = [] # Array que guarda cada objeto recortado da imagem

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
		img.binarized = threshold(img.image)
		img.pb = cv2.cvtColor(img.image, cv2.COLOR_BGR2GRAY)

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
		if np.abs(b-a) > 60:
			return False
	return True

def crop_minAreaRect(src, rect):
	 # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)

	# cv2.imshow('teste', out)
    # cv2.waitKey(0)
    return out

def cropBoundingRect(src, x, y, w, h):

	img_crop = src[y:y+h, x:x+w]

	return img_crop


def segmentation(images):
	for img in images:
		contours, hier = cv2.findContours(img.binarized,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		# print img.name
		# cv2.imshow('teste', img.image)
		# cv2.waitKey(0)

		for cnt in contours:
			if cv2.contourArea(cnt) < 500:
		  		continue
			x, y, w, h = cv2.boundingRect(cnt)

			img_croped = cropBoundingRect(img.pb, x, y, w, h)
			img_crop_bin = cropBoundingRect(img.binarized, x, y, w, h)
			img.objects.append(img_croped)
			img.objects_bin.append(img_crop_bin)

			#Pra testar deteccao de objetos, descomentar linhas abaixo
			# box = cv2.boxPoints(rect)
			# box = np.int0(box)
			# cv2.drawContours(img.image,[box],0,(255,255,255),1)


		# print img.name
		# cv2.imshow('teste', img.image)
		# cv2.waitKey(0)

def test_cropping(images):
	for img in images:
		print img.name

		cv2.imshow('teste', img.image)
		cv2.waitKey(0)

		for obj in img.objects:
			cv2.imshow('teste', obj)
			cv2.waitKey(0)


def classification (images):
	nomes = []
	huMoments = []
	for img in images:
		for i in range(len(img.objects)):
			obj = img.objects[i]
			seg_binarization = cv2.adaptiveThreshold(obj,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
			seg_binarization = cv2.bitwise_and( seg_binarization, img.objects_bin[i])
			seg_binarization = cv2.bitwise_or( seg_binarization, cv2.bitwise_not(img.objects_bin[i]))
			huMoments.append(calculaMomentosHu(seg_binarization))
			nomes.append(img.name)

	huMoments = np.array(huMoments)
	huMoments = normalize(huMoments)
	models = KMeans(n_clusters=7)
	modelsOutput = models.fit(huMoments)
	labels = models.predict(huMoments)
	cent = modelsOutput.cluster_centers_

	print("Score da silhueta"+ skm.silhouette_score(huMoments, labels))

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection='3d')
	# if(DEBUG):
	# verificaMelhorGrafico(huMoments, labels, cent)
	ax.scatter(huMoments[:,4], huMoments[:,6], huMoments[:,5], c=labels)
	ax.scatter(cent[:,4], cent[:,6], cent[:,5], c='red', marker="*")
	plt.show()

	#
	# for img in images:
	# 	for i in range(len(img.objects)):
	# 		img.classificationObjects.append(labels[i])
	# 	print(img.name)
	# 	print(img.classificationObjects)
	# 	print("**********************")

def calculaMomentosHu(image):
	huMoment = cv2.HuMoments(cv2.moments(image)).flatten()
	huMoment = map(lambda hu: -1 * np.sign(hu) * np.log10(np.abs(hu)), huMoment)
	return huMoment

def verificaMelhorGrafico(huMoments, labels, cent):
	for i in range(0,3):
		for j in range(0,3):
			print(i,j)
			plt.scatter(huMoments[:,i], huMoments[:,j], c=labels)
			plt.scatter(cent[:,i], cent[:,j], c='red', marker="*")
			plt.show()


#====================================================
#main
#====================================================

#Loads Database
db = DB('./images')
images = db.getData()
print("Finished Loading Database")

#Segmentation
binarization(images)
print("Finished Binarization")

# test_cropping(images[:2])
segmentation(images)
#Printing test
# for img in images:
	# print(img.name, img.classification)

#Classification
classification(images)
