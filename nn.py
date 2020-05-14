### Science is a differential equation. Religion is a boundary condition. — Alan Turing

import numpy as np
from PIL import Image
from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as plt
import scipy.io
import scipy.optimize as optmz

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
    
    def forwardProp(self,X,y,Theta1,Theta2,_lambda):
        m = X.shape[0]
        X = np.column_stack((np.ones((m,1)),X))
        a_1 = self.sigmoid(Theta1.dot(X.T))
        a_1 = np.vstack(((np.ones((1,a_1.shape[1]))),a_1))
        a_2 = self.sigmoid(Theta2.dot(a_1))
        return a_2,a_1,X

    def cost_grad(self,X,y,Theta1,Theta2,_lambda):
        m = X.shape[0]
        a_2,a_1,X = self.forwardProp(X,y,Theta1,Theta2,_lambda)
        left = sum(sum(-y.T * np.log(a_2)))
        right = sum(sum((1-y).T * np.log(1 - a_2)))
        
        # Cost with out regularization 
        J = 1/m * (left -right)
        
        # Cost with regularization
        cost_regu_1 = np.sum(Theta1[1:]**2)
        cost_regu_2 = np.sum(Theta2[1:]**2)
        cost_regu = (_lambda/(2*m)) * (cost_regu_1 +cost_regu_2)
        J = J + cost_regu

        # Backpropagation without regularization 
        delta_3 = a_2 - y.T
        delta_2_ = Theta2.T.dot(delta_3) * a_1 * (1-a_1)
        delta_2 = np.delete(delta_2_,0,0)

        Delta_1 = delta_2.dot(X)
        Delta_2 = delta_3.dot(a_1.T) 

        Theta1_grad = Delta_1/m
        Theta2_grad = Delta_2/m 

        # Backpropagation with regularization
        Theta1_grad[1:] = Theta1_grad[1:] + (_lambda/m) * Theta1[1:]
        Theta2_grad[1:] = Theta2_grad[1:] + (_lambda/m) * Theta2[1:]
        Theta1_grad = Theta1_grad.reshape(10025)
        Theta2_grad = Theta2_grad.reshape(260)
        grad =  np.append(Theta1_grad,Theta2_grad)
        return  J, grad   
    def grad_func(self,X,y,Theta1,Theta2,_lambda):
        m = X.shape[0]
        a_2,a_1,X = self.forwardProp(X,y,Theta1,Theta2,_lambda)
        # Backpropagation without regularization 
        delta_3 = a_2 - y.T
        delta_2_ = Theta2.T.dot(delta_3) * a_1 * (1-a_1)
        delta_2 = np.delete(delta_2_,0,0)

        Delta_1 = delta_2.dot(X)
        Delta_2 = delta_3.dot(a_1.T) 

        Theta1_grad = Delta_1/m
        Theta2_grad = Delta_2/m 

        # Backpropagation with regularization
        Theta1_grad[1:] = Theta1_grad[1:] + (_lambda/m) * Theta1[1:]
        Theta2_grad[1:] = Theta2_grad[1:] + (_lambda/m) * Theta2[1:]
        Theta1_grad = Theta1_grad.reshape(10025)
        Theta2_grad = Theta2_grad.reshape(260)
        grad =  np.append(Theta1_grad,Theta2_grad)

        return grad

    def sigmoid(self,z):
        return (1./(1. + np.exp(-z)))
    
    #sigmoidGradient
    def sigmoidGradient(self,z):
        return (1./(1. + np.exp(-z))) * (1 - (1./(1. + np.exp(-z))))
    
    #Random Initialize Weights
    def randInitializeWeights(self,L_in,L_out):
        epsilon = 0.12
        w = np.random.uniform(0,1,(L_out, 1 + L_in)) * 2 * epsilon - epsilon
        return w

neuralNet = nn('train-images-idx3-ubyte','train-labels-idx1-ubyte')
X,y,Theta1,Theta2 = neuralNet.load_matlab_data()
# X,y = neuralNet.load_data()
#Display random image of digits
i = 0
y_matrix = np.zeros((len(y),11))
for i in range(len(y)):
    y_matrix[i] = np.eye(1,11,k=y[i,0],dtype='int')
y_matrix = np.delete(y_matrix,0,1)
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

# parameters
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   

cost = neuralNet.cost_grad(X,y_matrix,Theta1,Theta2,1)
val = neuralNet.sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))

#Random Initialize Weights
intial_theta1 = neuralNet.randInitializeWeights(input_layer_size,hidden_layer_size)
intial_theta2 = neuralNet.randInitializeWeights(hidden_layer_size,num_labels)

initial_nn_params = np.append(intial_theta1.reshape(10025),intial_theta2.reshape(260))

x0 = initial_nn_params
myargs = ((X),(y_matrix))
lambda_ = 0.1
theta = optmz.fmin_bfgs(neuralNet.cost_grad(X,y_matrix,intial_theta1,intial_theta2,lambda_), x0, fprime = neuralNet.grad_func(X,y_matrix,intial_theta1,intial_theta2,lambda_), epsilon=lambda_,maxiter=500)
print(theta) 
print(val)