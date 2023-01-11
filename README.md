# Non-local-mean-filter

NL-means filter: weighted average of all the voxel intensities in the image 𝑢


![image](https://user-images.githubusercontent.com/118721799/211859414-0c70206d-f3ba-468c-8c5c-496dd7fe1249.png)


reference) Coupé, Pierrick, et al. "An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images." IEEE transactions on medical imaging 27.4 (2008): 425-441.

# Data Generation

- Before going on the Non-local-mean-filter phase, execute dataset.py to make your data noisy


      python dataset.py
        

#Non-local-mean-filter

- This code if for the non-local-filter


      python NonLocalMeanFilter.py
