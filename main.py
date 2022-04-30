import cv2
import matplotlib.pyplot as plt
import numpy as np

imagem = cv2.imread('.\images\my-image.png') 

(canalAzul, canalVerde, canalVermelho) = cv2.split(imagem)

cv2.imshow('canalVerde', canalVerde)
cv2.imshow('canalAzul', canalAzul)
cv2.imshow('canalVermelho', canalVermelho)
cv2.waitKey(0)

scale = 1
delta = 0
ddepth = cv2.CV_16S

src = cv2.GaussianBlur(canalVerde, (3, 3), 0)

grad_x = cv2.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

grad_y = cv2.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

cv2.imshow('Verde', grad)
cv2.waitKey(0)
