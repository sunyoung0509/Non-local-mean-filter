#final 

import cv2
import matplotlib.pyplot as plt
import numpy as np
from functools import partial, reduce 
import skimage.color as color
#functools: 고차 함수를 위한 것
#partial: 원래 있는 함수의 새 버전을 채우기 위해 사용된다.
#reduce(lambda arg: expression, list): 리스트의 원소에 누적으로 함수 적용(여러 개의 데이터를 대상으로 주로 누적 집계) 

data_dir = '' #write your directory
img_n = cv2.imread(data_dir + "/noisy.png")
plt.axis('off')
img_n = color.rgb2lab(img_n)
img_n = img_n[:, :, 0]
plt.imshow(img_n, cmap = 'gray')

#PSNR: 화질의 손실 정보 평가, MSE값이 작을 수록 PSNR이 높다 
def PSNR(original, noisy, peak=100):
    mse = np.mean((original - noisy)**2)
    return 10*np.log10(peak**2 / mse)

def pixel_value(value, pixel_window, h2, Nw):
    patch_window, median_diff = value
    diff = np.sum((pixel_window - patch_window)**2)

    diff -= median_diff
    w = np.exp(-diff / (h2 * Nw))
    
    nr = patch_window.shape[0]
    nc = patch_window.shape[1]

    intensity = w * patch_window[nr // 2, nc // 2]
    return intensity, w

#Non Linear Mean filter 
def NLM(img, n_big, n_small, h):
   
    new_n = np.zeros_like(img)
    
    Nw = (2 * n_small + 1) ** 2
    h2 = h * h
    n_cols = img.shape[1]
    n_rows = img.shape[0]
    
    #big patch의 좌표(r, c) 차이 계산
    D = range(-n_big, n_big + 1)
    big_diff = [(r, c) for r in D for c in D if not (r == 0 and c == 0)]
    
    #small patch의 좌표 차이 계산
    small_rows, small_cols = np.indices((2 * n_small +1, 2 * n_small +1)) - n_small

    padding = n_big + n_small
    n_padded = np.pad(img, padding, mode = 'reflect')
        
    for r in range(padding, padding + n_rows):
        for c in range(padding, padding + n_cols):
            pixel_window = n_padded[small_rows + r, small_cols + c]
            
            #patch window list 구성
            windows = [n_padded[small_rows + r + d[0], small_cols + c + d[1]] for d in big_diff]
            
            #median_difference list 구성
            median_diffs = [(n_padded[r, c] - n_padded[r + d[0], c + d[1]]) for d in big_diff]
            
            distance_map = partial(pixel_value, pixel_window = pixel_window, h2=h2, Nw=Nw)
            distances = map(distance_map, zip(windows, median_diffs))
            
            total_c, total_w = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]), distances)
            
            new_n[r - padding, c - padding] = total_c / total_w 
            
    return new_n
        
std = 20
denoised_naive = []
denoised_image = NLM(img_n, n_big=7, n_small=1, h=10*std)
denoised_naive.append(denoised_image)

Titles = ['noisy image', 'denoised image']
images = [img_n, denoised_image]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i])
    plt.title(Titles[i])
plt.show()

#opencv 내의 내장함수를 이용해서 non local mean filter만드는 방법 (아래 주석 처리 지우고 해보세요)

# denoised_image = cv2.fastNlMeansDenoisingColored(img_n, None, 15, 15, 5, 10)

# Titles = ['noisy', 'denoised']
# images = [img_n, denoised_image]

# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i])
#     plt.title(Titles[i])
# plt.show()