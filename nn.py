import numpy as np
from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as plt

X, y = loadlocal_mnist(
        images_path='/Users/de/Documents/project/train-images-idx3-ubyte', 
        labels_path='/Users/de/Documents/project/train-labels-idx1-ubyte')

first_image = np.array(X[0], dtype='float')
pixel = first_image.reshape((28,28))
plt.imshow(pixel,cmap = 'gray')
plt.show()