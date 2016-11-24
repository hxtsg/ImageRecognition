import pickle
import random
import numpy as np
from PIL import Image
import cv2


class KNN():
    def __init__(self):
        self.label_names = None
    def LOAD_CIFAR_DATA(self, filename):
        with open(filename,'rb') as f:
            dataset = pickle.load( f )
            X = dataset['data']
            Y = dataset['labels']
            X = np.reshape( X, ( 10000, 3, 32, 32 ) )
            Y = np.array( Y )
            return X, Y

    def LOAD_CIFAR_LABELS(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load( f )
            self.label_names =  obj['label_names']



def main():
    knn = KNN()
    imgFileName = './cifar-10-batches-py/data_batch_1'
    labelFileName = './cifar-10-batches-py/batches.meta'
    knn.LOAD_CIFAR_LABELS( filename = labelFileName )
    X,Y = knn.LOAD_CIFAR_DATA( filename = imgFileName )
    example_nums = X.shape[0]
    for i in range( example_nums ):
        img = X[ i ]
        img_merged = cv2.merge( [ img[0],img[1],img[2] ] )
        cv2.imshow( "Image", img_merged )
        print knn.label_names[Y[i]]
        cv2.waitKey(0)



if __name__ == '__main__':
    main()