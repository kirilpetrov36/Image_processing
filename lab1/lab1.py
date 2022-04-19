import numpy as np
import cv2
from matplotlib import pyplot as plt

def sobelOperator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 2):
        for j in range(1, size[1] - 2):
            gx = (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1]) - (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1])
            gy = (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1]) - (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container

girl_sobel = cv2.cvtColor(cv2.imread('girl.png'), cv2.COLOR_BGR2GRAY)
girl_image = cv2.imread('girl.png')
girl_sobel = sobelOperator(girl_sobel)
girl_sobel = cv2.cvtColor(girl_sobel, cv2.COLOR_GRAY2RGB)

#plt.imshow(img)
fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(girl_image)
plt.axis('off')
plt.title("Before Sobel")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(girl_sobel)
plt.axis('off')
plt.title("After Sobel")
plt.show()

cube_sobel = cv2.cvtColor(cv2.imread('cube.png'), cv2.COLOR_BGR2GRAY)
cube_image = cv2.imread('cube.png')
cube_sobel = sobelOperator(cube_sobel)
cube_sobel = cv2.cvtColor(cube_sobel, cv2.COLOR_GRAY2RGB)

#plt.imshow(img)
fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(cube_image)
plt.axis('off')
plt.title("Before Sobel")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(cube_sobel)
plt.axis('off')
plt.title("After Sobel")
plt.show()

def RobertsOperator(img):
    container = np.copy(img)
    size = container.shape
    print(type(img[0][0]))
    for i in range(0, size[0] - 2):
        for j in range(0, size[1] - 2):
            gx = int(img[i+1][j+1]) - int(img[i][j])
            gy = int(img[i+1][j]) - int(img[i][j+1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))       
    return container

girl_roberts = RobertsOperator(cv2.cvtColor(girl_image, cv2.COLOR_BGR2GRAY))
girl_roberts = cv2.cvtColor(girl_roberts, cv2.COLOR_GRAY2RGB)

#plt.imshow(img)
fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(girl_image)
plt.axis('off')
plt.title("Before Roberts")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(girl_roberts)
plt.axis('off')
plt.title("After Roberts")
plt.show()

cube_roberts = RobertsOperator(cv2.cvtColor(cube_image, cv2.COLOR_BGR2GRAY))
cube_roberts = cv2.cvtColor(cube_roberts, cv2.COLOR_GRAY2RGB)

#plt.imshow(img)
fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(cube_image)
plt.axis('off')
plt.title("Before Roberts")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(cube_roberts)
plt.axis('off')
plt.title("After Roberts")
plt.show()

#plt.imshow(img)
fig = plt.figure(figsize=(20, 50))
rows = 1
columns = 3

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(girl_image)
plt.axis('off')
plt.title("Girl image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(girl_sobel)
plt.axis('off')
plt.title("After Sobel")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(girl_roberts)
plt.axis('off')
plt.title("After Roberts")
plt.show()

#plt.imshow(img)
fig = plt.figure(figsize=(20, 50))
rows = 1
columns = 3

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(cube_image)
plt.axis('off')
plt.title("Girl image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(cube_sobel)
plt.axis('off')
plt.title("After Sobel")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(cube_roberts)
plt.axis('off')
plt.title("After Roberts")
plt.show()

