import numpy as np
import scipy.io as sio
from scipy import misc
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb
RootDir="D:/PhD"
PrefixToDir={"fd":"father-dau","fs":"father-son","md":"mother-dau","ms":"mother-son"}

def FlipAugment(X,Y):
    s=X.shape
    MoreX=[]
    MoreY=[]
    for i in range(s[0]):
        pImg=X[i][0]
        cImg=X[i][1]
        kOut=Y[i]
        MoreX.append([pImg,cImg])
        MoreY.append(kOut)
        MoreX.append([np.fliplr(pImg),cImg])
        MoreY.append(kOut)
        MoreX.append([pImg,np.fliplr(cImg)])
        MoreY.append(kOut)
        MoreX.append([np.fliplr(pImg),np.fliplr(cImg)])
        MoreY.append(kOut)
    return (np.array(MoreX),np.array(MoreY))

def LoadFoldK(KinSet,KinShip,Fold,ValidSplit):
    meta=sio.loadmat(RootDir+"/"+KinSet+"/meta_data/"+KinShip+"_pairs.mat")
    pairs=meta['pairs']
    TrainX=[]
    TrainY=[]
    ValidX=[]
    ValidY=[]
    TestX=[]
    TestY=[]
    TrainN=0
    pDir=RootDir+"/"+KinSet+"/images/"+PrefixToDir[KinShip]+"/"
    for p in pairs:
        pImg=misc.imread(pDir+p[2][0])/255.0
        cImg=misc.imread(pDir+p[3][0])/255.0
        if p[0][0][0]==Fold:
            TestX.append([pImg,cImg])
            TestY.append([p[1][0][0]])
        else:
            ValidN=int(TrainN*ValidSplit)
            if ValidN>=1:
                ValidX.append([pImg,cImg])
                ValidY.append([p[1][0][0]])
                TrainN=0
            else:
                TrainX.append([pImg,cImg])
                TrainY.append([p[1][0][0]])
                TrainN=TrainN+1
    return (np.array(TrainX),np.array(TrainY),np.array(ValidX),np.array(ValidY),np.array(TestX),np.array(TestY))

def LoadFold(Fold,ValidSplit):
    KinSets=[("fs","KinFaceW-I"),("fd","KinFaceW-I"),("ms","KinFaceW-I"),("md","KinFaceW-I"),("fs","KinFaceW-II"),("fd","KinFaceW-II"),("ms","KinFaceW-II"),("md","KinFaceW-II")]
    Data=[]
    for (KinShip,KinSet) in KinSets:
        Data.append(LoadFoldK(KinSet,KinShip,Fold,ValidSplit))
    X0=np.concatenate((Data[0][0],Data[1][0],Data[2][0],Data[3][0],Data[4][0],Data[5][0],Data[6][0],Data[7][0]),axis=0)
    Y0=np.concatenate((Data[0][1],Data[1][1],Data[2][1],Data[3][1],Data[4][1],Data[5][1],Data[6][1],Data[7][1]),axis=0)
    X1=np.concatenate((Data[0][2],Data[1][2],Data[2][2],Data[3][2],Data[4][2],Data[5][2],Data[6][2],Data[7][2]),axis=0)
    Y1=np.concatenate((Data[0][3],Data[1][3],Data[2][3],Data[3][3],Data[4][3],Data[5][3],Data[6][3],Data[7][3]),axis=0)
    X2={"fs-I":Data[0][4],"fd-I":Data[1][4],"ms-I":Data[2][4],"md-I":Data[3][4],"fs-II":Data[4][4],"fd-II":Data[5][4],"ms-II":Data[6][4],"md-II":Data[7][4]}
    Y2={"fs-I":Data[0][5],"fd-I":Data[1][5],"ms-I":Data[2][5],"md-I":Data[3][5],"fs-II":Data[4][5],"fd-II":Data[5][5],"ms-II":Data[6][5],"md-II":Data[7][5]}
    return (X0,Y0,X1,Y1,X2,Y2)

def SavePairs(P,C,K,D):
    N=K.shape[0]
    PID=1
    NID=1
    for i in range(N):
        if K[i]==1:
            misc.imsave("./%s/Positive/%05d_1.jpg"%(D,PID),P[i])
            misc.imsave("./%s/Positive/%05d_2.jpg"%(D,PID),C[i])
            PID=PID+1
        else:
            misc.imsave("./%s/Negative/%05d_1.jpg"%(D,NID),P[i])
            misc.imsave("./%s/Negative/%05d_2.jpg"%(D,NID),C[i])
            NID=NID+1
