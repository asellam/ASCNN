# ASCNN
All-subsets Siamese Convolutional Neural Network for Kinship Verification.

# Prerequisites
1. Python
1. Tensorflow (pip install tensorflow-gpu)
1. Numpy (pip install numpy)
1. Scipy (pip install scipy)
1. Keras 2.0

# How To Use
1. Run from Command Line Prompt: SiameseCNN.py (FoldNumber 1..5)
 OR
1. Run using the batch file: SiameseCNN.bat, this will run the code with fold number 1 to 5

The results will be outputed to a csv file (scnn_results.csv)

# Important
Before running the code make sure to:
1. Download the KinFaceW data-sets from these links: http://www.kinfacew.com/dataset/KinFaceW-I.zip
http://www.kinfacew.com/dataset/KinFaceW-II.zip
1. Unzip the two files somewhere in your hard disk
1. Go to LoadData.py and set RootDir to the path of the directory containing the two data sets: KinFaceW-I and KinFaceW-II
