#----------------------------------------
#- Siamese CNN for Kinship Verification -
#----------------------------------------
#- By: Abdellah SELLAM                  -
#-     Hamid AZZOUNE                    -
#----------------------------------------
#- Created: 2018-07-02                  -
#- Last Update: 2018-09-08              -
#----------------------------------------

from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import sys
import csv
import os.path as path
#Our DataSet I/O Routines
from LoadData import LoadFold
from LoadData import SavePairs
from matplotlib import pyplot as plt
from scipy import misc

#A function that initializes layers weigths
def W_init(shape,name=None):
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

#A function that initializes layers biases
def b_init(shape,name=None):
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

#Get the fold number from command line arguments
F=int(sys.argv[1])

#Percentage of data used to validate
ValidSplit=0.2

#Loads data of the Fold #F
#X0: Training Inputs
#Y0: Training Targets
#X1: Validation Inputs
#Y1: Validation Targets
#X2: Test Inputs
#Y2: Test Targets
(X0,Y0,X2,Y2,X1,Y1)=LoadFold(F,ValidSplit)
#Dimensions of the training inputs array
S0=X0.shape
#Inputs Array's Shape
in_shape=(S0[2],S0[3],S0[4])
#Input Layer of the first (left) ConvNet (Images of parents)
left_input = Input(in_shape)
#Input Layer of the second (right) ConvNet (Images of children)
right_input = Input(in_shape)
#The definition of the convnet to be used for left and right inputs
convnet = Sequential()
#16 Convolutions of 13x13 size and 1x1 stride and ReLU activation
C1=Conv2D(16,(13,13),activation='relu',input_shape=in_shape,padding="same",
            kernel_regularizer=l2(0.001),kernel_initializer=W_init,bias_initializer=b_init)
convnet.add(C1)
#Max Pooling of 2x2 size and 2x2 stride
PL1=MaxPooling2D()
convnet.add(PL1)

#64 Convolutions of 5x5 size and 1x1 stride and ReLU activation
C2=Conv2D(64,(5,5),activation='relu',padding="same",
            kernel_regularizer=l2(0.001),kernel_initializer=W_init,bias_initializer=b_init)
convnet.add(C2)
#Max Pooling of 2x2 size and 2x2 stride
PL2=MaxPooling2D()
convnet.add(PL2)

#128 Convolutions of 3x3 size and 1x1 stride and ReLU activation
C3=Conv2D(128,(3,3),activation='relu',padding="same",
            kernel_regularizer=l2(0.001),kernel_initializer=W_init,bias_initializer=b_init)
convnet.add(C3)
#Max Pooling of 2x2 size and 2x2 stride
PL3=MaxPooling2D()
convnet.add(PL3)

#This layer transform a 3D volume to a 1D vector by a flattenning operation
convnet.add(Flatten())
#Left input (Parent Image) encoded by the convnet into a 1D feature vector
encoded_l = convnet(left_input)
#Right input (Child Image) encoded by the convnet into a 1D feature vector
encoded_r = convnet(right_input)
#Define the L1 lambda function to be used in merging encoded inputs
L1_distance = lambda x: K.abs(x[0]-x[1])
#Merge the two encoded inputs (1D feature vectors) using L1 norm
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
#A dense layer with Sigmoid activation applied to the merged vectors to
# compute kinship output
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
#The Siamese CNN defined as model taking as input two facial images and
# outputting the kinship output computed by applying the final dense layer
# to the merged vectors extracted by the two identical convnets sharing the
# same set of weights!
SiameseCNN = Model(input=[left_input,right_input],output=prediction)
#SGD optimizer with a learning rate of 1.0
optimizer = SGD(0.1)
#Compile the Keras Model
SiameseCNN.compile(loss="mean_squared_error",optimizer=optimizer,metrics=["accuracy"])
#print the number of Training/Validation samples
print("Training on",X0.shape[0],"samples, validation on",X2.shape[0],"samples")
#Training the Siamese CNN with a goal loss
#Current Epoch Counter
Epochs=0
#Maximum neumber of epochs
MaxEpochs=200
#Goal loss on training data
TrainGoal=0.85
#Goal loss on validation data
ValidGoal=0.80
#Current accuracy on training data
TrainAccu=0.0
#Current accuracy on validation data
ValidAccu=0.0
#Conitnue as long as not reached maximum number of epochs and one of the loss
# values (for train/validation data) did not reach its goal value
while (Epochs<MaxEpochs)and((TrainAccu<TrainGoal)or(ValidAccu<ValidGoal)):
    #A single epoch training
    SiameseCNN.fit([X0[:,0,:,:,:],X0[:,1,:,:,:]],Y0,validation_data=([X2[:,0,:,:,:],X2[:,1,:,:,:]],Y2),epochs=1,verbose=0,batch_size=100)
    #Compute Training Data's loss/accuracy
    TrainE=SiameseCNN.evaluate(x=[X0[:,0,:,:,:],X0[:,1,:,:,:]],y=Y0,verbose=0)
    #Compute Validation Data's loss/accuracy
    ValidE=SiameseCNN.evaluate(x=[X2[:,0,:,:,:],X2[:,1,:,:,:]],y=Y2,verbose=0)
    #Training Data's accuracy
    TrainAccu=TrainE[1]
    #Validation Data's accuracy
    ValidAccu=ValidE[1]
    #Increment the epoch's counter
    Epochs=Epochs+1
    #Display Epoch, loss, accuracy ...
    print("Epoch: %d/%d"%(Epochs,MaxEpochs))
    print("loss: %.04f - accuracy: %.04f - valid_loss: %.04f - valid_accuracy: %.04f"%(TrainE[0],TrainE[1],ValidE[0],ValidE[1]))


#Save accuracies for this Fold as a Comma-Separated-Values file
#Read old accuracies if they exists
FileName="scnn_results.csv"
if(path.isfile(FileName)):
    csvr=open(FileName,"r")
    rows=csv.reader(csvr,delimiter=';')
    csvd=[row for row in rows]
    data={"fs-I":[csvd[i][1] for i in range(1,6)],
          "fd-I":[csvd[i][2] for i in range(1,6)],
          "ms-I":[csvd[i][3] for i in range(1,6)],
          "md-I":[csvd[i][4] for i in range(1,6)],
          "fs-II":[csvd[i][5] for i in range(1,6)],
          "fd-II":[csvd[i][6] for i in range(1,6)],
          "ms-II":[csvd[i][7] for i in range(1,6)],
          "md-II":[csvd[i][8] for i in range(1,6)]}
    csvr.close()
else:
    data={"fs-I":["0.00","0.00","0.00","0.00","0.00"],"fd-I":["0.00","0.00","0.00","0.00","0.00"],
          "ms-I":["0.00","0.00","0.00","0.00","0.00"],"md-I":["0.00","0.00","0.00","0.00","0.00"],
          "fs-II":["0.00","0.00","0.00","0.00","0.00"],"fd-II":["0.00","0.00","0.00","0.00","0.00"],
          "ms-II":["0.00","0.00","0.00","0.00","0.00"],"md-II":["0.00","0.00","0.00","0.00","0.00"]}

#Check if new results are better
vote=0.0
for KinSet in ["md-II"]:
    TestE=SiameseCNN.evaluate(x=[X1[KinSet][:,0,:,:,:],X1[KinSet][:,1,:,:,:]],y=Y1[KinSet],verbose=0)
    vote=vote+TestE[1]-float(data[KinSet][F-1])
    
#if new results are better then save this mode
if vote>0.0:
    SiameseCNN.save("ASCNN_%d.h5"%(F))

#if new results are better then save them
for KinSet in X1:
    TestE=SiameseCNN.evaluate(x=[X1[KinSet][:,0,:,:,:],X1[KinSet][:,1,:,:,:]],y=Y1[KinSet],verbose=0)
    print("Fold:",F)
    print("DataSet:",KinSet)
    print("Accuracy:",TestE[1])
    if vote>0.0:
        data[KinSet][F-1]="%.04f"%(TestE[1])

#Write saved data to disc as a Comma-Separated-Values file
csvw=open(FileName,"w")
csvw.write("Fold;fs-I;fd-I;ms-I;md-I;fs-II;fd-II;ms-II;md-II\n")
for F in range(5):
    csvw.write("%d;%s;%s;%s;%s;%s;%s;%s;%s\n"%(F+1,data["fs-I"][F],data["fd-I"][F],data["ms-I"][F],data["md-I"][F],data["fs-II"][F],data["fd-II"][F],data["ms-II"][F],data["md-II"][F]))
csvw.close()

# Visualize Hidden Conv Layers' outputs
def ViewLayerOutput(S,L,Title):
    LayerF = K.function(convnet.inputs,[L.output])
    O1=LayerF([np.array([S[0]])])
    O2=LayerF([np.array([S[1]])])
    m=min(max(O1[0].shape[3],O2[0].shape[3]),8)
    fig=plt.figure(figsize=(2,m+1))
    fig.canvas.set_window_title(Title)
    fig.canvas._master.geometry('1024x512+171+128')
    fig.add_subplot(2,m+1,1)
    plt.imshow(S[0])
    for i in range(m):
        fig.add_subplot(2,m+1,i+2)
        plt.imshow(O1[0][0,:,:,i],cmap=plt.get_cmap('gray'))
    fig.add_subplot(2,m+1,m+2)
    plt.imshow(S[1])
    for i in range(m):
        fig.add_subplot(2,m+1,m+3+i)
        plt.imshow(O2[0][0,:,:,i],cmap=plt.get_cmap('gray'))
    plt.show()

# Saves Hidden Conv Layers' outputs
def SaveLayerOutput(S,L,N,Name):
    LayerF = K.function(convnet.inputs,[L.output])
    O1=LayerF([np.array([S[0]])])
    O2=LayerF([np.array([S[1]])])
    misc.imsave("LayersDisplayed/%03d_1.jpg"%(N),S[0])
    misc.imsave("LayersDisplayed/%03d_2.jpg"%(N),S[1])
    for i in range(O1[0].shape[3]):
        misc.imsave("LayersDisplayed/%03d_1_%s_%03d.jpg"%(N,Name,i+1),O1[0][0,:,:,i])
    for i in range(O2[0].shape[3]):
        misc.imsave("LayersDisplayed/%03d_2_%s_%03d.jpg"%(N,Name,i+1),O2[0][0,:,:,i])

#Saving hidden conv layers outputs as gray-scale image to a directory named:
        # 'LayersDisplayed', the directory must be created in advance
for kin in X1:
    x1=X1[kin]
    N1=x1.shape[0]
    for s in range(N1):
        SaveLayerOutput(x1[s],C1,s,"Conv2d1")#First Conv2D Layer
        SaveLayerOutput(x1[s],C2,s,"Conv2d2")#Second Conv2D Layer
        SaveLayerOutput(x1[s],C3,s,"Conv2d3")#Third Conv2D Layer
