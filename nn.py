import numpy as np
from PIL import Image
from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as plt
import scipy.io

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
    
    def load_matlab_data(self):
        data = scipy.io.loadmat('ex4data1.mat')
        theta = scipy.io.loadmat('ex4weights.mat')
        X = data['X']
        y = data['y']
        Theta1 = theta['Theta1']
        Theta2 = theta['Theta2']
        return X,y,Theta1,Theta2

    def cost_grad(self,X,y,Theta1,Theta2):
        m = X.shape[0]
        X = np.column_stack((np.ones((m,1)),X))
        a_1 = self.sigmoid(Theta1.dot(X.T))
        a_1 = np.vstack((a_1,(np.ones((1,a_1.shape[1])))))
        b_2 = self.sigmoid(Theta2.dot(a_1))
        return 1       

    def sigmoid(self,z):
        return (1/(1 + np.exp(-z)))

neuralNet = nn('train-images-idx3-ubyte','train-labels-idx1-ubyte')
X,y,Theta1,Theta2 = neuralNet.load_matlab_data()
# X,y = neuralNet.load_data()
#Display random image of digits
fig=plt.figure(figsize=(5, 5))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    rand = np.random.randint(5000)
    first_image = np.array(X[rand], dtype='float')
    pixel = first_image.reshape((20,20))
    fig.add_subplot(rows, columns,i)
    plt.imshow(pixel,cmap = 'gray')
plt.show()