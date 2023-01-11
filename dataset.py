#%%

import cv2
import matplotlib.pyplot as plt
import numpy as np

data_dir = '' #write your directory
img = cv2.imread(data_dir + "/gt_EX4249_one.png")
img2 = cv2.imread(data_dir + "/gt_EX4249_one.png", cv2.IMREAD_GRAYSCALE)
img3 = plt.imread(data_dir + "/gt_EX4249_one.png")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(img2)
plt.imshow(img2, cmap = 'gray')
plt.imshow(img3)

img_shape = img3.shape

std = 20

def make_noise(std, gray): #std: 잡음의 크기
    height, width = gray.shape
    img_noise = np.zeros((height, width), dtype = np.float64)
    
    #가우시안 노이즈 만들기 
    for i in range(height):
        for j in range(width):
            make_noise = np.random.normal() #random 함수 이용해서 노이즈 만들기
            set_noise = std * make_noise
            img_noise[i][j] = gray[i][j] + set_noise #set_noise를 더해주어 원본 영상에 노이즈 추가 
    return img_noise

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
img_noise = make_noise(std, gray)

img_noisy = plt.imshow(np.expand_dims(img_noise, axis=0).reshape(248, 410, 1), cmap='gray')
plt.axis('off') #x,y축 모두 없애기
plt.savefig("noisy.png", bbox_inches='tight')
plt.show()
print(img_noise.shape)
