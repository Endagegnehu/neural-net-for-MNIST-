import numpy as np
from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as plt
class nn():

    def __init__(self, image_path,labels_path):
        self.image_path = image_path
        self.labels_path = labels_path
    
    #Load local mnist data.   
    def load_data(self):
        X, y = loadlocal_mnist(
            images_path=self.image_path, 
            labels_path=self.labels_path)
        return X,y
neuralNet = nn('/Users/de/Documents/project/train-images-idx3-ubyte','/Users/de/Documents/project/train-labels-idx1-ubyte')
X,y = neuralNet.load_data()

#Display some image of digits  
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    first_image = np.array(X[i], dtype='float')
    pixel = first_image.reshape((28,28))
    fig.add_subplot(rows, columns, i)
    plt.imshow(pixel,cmap = 'gray')
plt.show()

