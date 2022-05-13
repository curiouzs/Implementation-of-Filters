# Implementation-of-Filters
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the necessary modules.

### Step2:
For performing smoothing operation on a image.

Average filter
```python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
```

Weighted average filter
```python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
```

Gaussian Blur
```python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
Median filter
median=cv2.medianBlur(image2,13)
```

### Step3:
For performing sharpening on a image.

Laplacian Kernel
```python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
```

Laplacian Operator
```python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
```

### Step4:
Display all the images with their respective filters.




## Program: 
### Developed By   :LOKESH KRISHNAA M
### Register Number:212220230030
</br>

### 1. Smoothing Filters

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("loki.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

```

i) Using Averaging Filter
```Python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()


```
ii) Using Weighted Averaging Filter
```Python

kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()



```
iii) Using Gaussian Filter
```Python

gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()


```

iv) Using Median Filter
```Python

median=cv2.medianBlur(image2,13)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()



```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python

kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()



```
ii) Using Laplacian Operator
```Python

laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()



```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter
</br>
![Screenshot (53)](https://user-images.githubusercontent.com/75234646/168294206-8821a86a-337d-41f5-b0c6-50b6068020e9.png)

</br>

ii) Using Weighted Averaging Filter
</br>
![Screenshot (54)](https://user-images.githubusercontent.com/75234646/168294161-804c2641-e4e8-4581-8ca2-46f92dc3a8a7.png)

</br>

iii) Using Gaussian Filter
</br>
![Screenshot (55)](https://user-images.githubusercontent.com/75234646/168294373-f501fc70-a3d3-4eb5-9bf7-6739fc24c214.png)

</br>

iv) Using Median Filter
</br>
![Screenshot (56)](https://user-images.githubusercontent.com/75234646/168294119-3b543aa8-c5d1-4b46-85ad-41b4d6804107.png)


</br>

### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal
</br>
![Screenshot (57)](https://user-images.githubusercontent.com/75234646/168294328-507c4035-a46b-42ba-a2d2-5682fb6effdc.png)

</br>

ii) Using Laplacian Operator
</br>
![Screenshot (62)](https://user-images.githubusercontent.com/75234646/168294259-1c27f6bb-8da6-4d16-9c51-b0566c0c14fa.png)

</br>

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
