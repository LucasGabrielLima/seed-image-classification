import cv2, os
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/class3/Emex_australis_06_binary_saida_esperada.jpg')
img = cv2.bitwise_not(img)

# mask = np.zeros(img.shape,np.uint8)
teste = img.copy()	
teste = cv2.cvtColor(teste, cv2.COLOR_BGR2GRAY);

contours, hier = cv2.findContours(teste,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	print('countour')
	cv2.drawContours(img,[cnt],0,(0,255, 255),3)




cv2.imshow('teste', img)
cv2.waitKey(0)

cv2.imshow('teste', teste)
cv2.waitKey(0)