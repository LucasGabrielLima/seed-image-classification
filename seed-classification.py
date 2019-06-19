import cv2, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
	kernel = np.ones((2,2),np.uint8)
	for img in images:
		print img.name

		huMoments = []
		cv2.imshow('teste', img.image)
		cv2.waitKey(0)
		for i in range(len(img.objects)):
			obj = img.objects[i]
			seg_binarization = cv2.adaptiveThreshold(obj,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
			seg_binarization = cv2.bitwise_and( seg_binarization, img.objects_bin[i])
			seg_binarization = cv2.bitwise_or( seg_binarization, cv2.bitwise_not(img.objects_bin[i]))
			# closing = cv2.morphologyEx(seg_binarization, cv2.MORPH_CLOSE, kernel)
			cv2.imshow('teste', seg_binarization)
			cv2.waitKey(0)
			# cv2.imshow('teste', seg_binarization)
			# cv2.waitKey(0)
			moments = cv2.moments(seg_binarization)
			# print("moments")
			# print(moments)
			huMoment = cv2.HuMoments(seg_binarization)
			# print(huMoments)
			# print(huMoments)

			# Normalizacao
			for i in range(0,7):
  				huMoment[i] = -1* np.sign(1.0, huMoment[i]) * np.log10(np.abs(huMoment[i]))

			huMoments.append(huMoment)

	df = pd.DataFrame()

	huMoments = np.array(huMoments)
	for i in range(len(huMoments[0])):
		df["HU{}".format(i+1)] = huMoments[:,i]

	df.to_csv("huMoments.csv", encoding='utf-8',index = False)

	data_set = pd.read_csv("HuMoments.csv", header=None, delimiter = ',')
	# print data_set.tail()

	X = data_set[range(3)]
	X_normalized = normalize(X)
	# print X
	#declara classificador e treina
	kmeans = KMeans(n_clusters=7, init = 'random', random_state=0, max_iter = 600)
	kmeans.fit(X_normalized)
	y_kmeans = kmeans.predict(X_normalized)

	#adiciono coluna com o resultado do kmeans
	data_set[8] = y_kmeans
	print data_set
	#silhouette_score = resultado entre -1 e 1.
	#-1 indica que a clusterizacao deu errado
	#0 indica overlapping cluster
	#1 cluster top

	print silhouette_score(X_normalized, kmeans.labels_, metric = 'euclidean')

	plt.scatter(X_normalized[:,0], X_normalized[:,1], c=y_kmeans, s=20, cmap='viridis')

	centers = kmeans.cluster_centers_
	plt.scatter(centers[:, 0], centers[:, 1], c='red', s=150, alpha=0.5);

	#tentativa para plotar k-means
	# pl.figure('3 Cluster K-Means')
	# pl.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
	# pl.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red', label = 'Centoids')

	# plt.xlabel('Dividend Yield')
	# plt.ylabel('Returns')
	# plt.title('6 Cluster K-Means')
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

segmentation(images)
# test_cropping(images[:2])
#Printing test
# for img in images:
	# print(img.name, img.classification)

#Classification
classification(images)
