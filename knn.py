import pickle
import random
import numpy as np
from PIL import Image
import cv2



class KNN():
    def __init__(self):  # initialization is also the step of training
        self.data = Data()
        self.data.Run()
        self.predict_Y = np.zeros( ( self.data.test_X.shape[0] ) )
        # self.predict_Y = np.zeros( (100) )
        self.K = 5


    def getMaxCnt(self, list):
        bucket = {}
        for value in list:
            if bucket.has_key( value ):
                bucket[ value ] += 1
            else:
                bucket[ value ] = 1
        rst_cnt = 0
        rst = -1
        for value in bucket.keys():
            if bucket[ value ] > rst_cnt:
                rst_cnt = bucket[ value ]
                rst = value
        return rst

    def predict(self):

        set_num = self.data.test_X.shape[0]
        # print set_num
        for i in range( set_num ):
            print i
            distance = np.sum(np.abs( self.data.X_input - self.data.test_X[ i, : ] ), axis=1)

            min_index_array = np.argsort( distance )
            vote_cnt = []
            for k in range(self.K):
                vote_cnt.append( self.data.Y_input[ min_index_array[ k ] ] )


            self.predict_Y[ i ] = self.getMaxCnt( vote_cnt )


        accuracy = np.mean( self.predict_Y == self.data.test_Y )
        print "The accuracy is: ", accuracy





class Data():
    def __init__(self):
        self.label_names = None


        self.X_input_list = []
        self.Y_input_list = []


        self.X_input = np.zeros( (60000, 3 * 32 * 32) )   # training data and labels
        self.Y_input = np.zeros( (60000), dtype=np.int32)

        self.test_X = None   # test data and labels
        self.test_Y = None
    def Run(self):

        labelFileName = './cifar-10-batches-py/batches.meta'
        self.LOAD_CIFAR_LABELS(filename=labelFileName)

        for i in range(1,6):
            imgFileName = './cifar-10-batches-py/data_batch_' + str( i )
            self.LOAD_CIFAR_DATA(imgFileName )

        imgFileName = './cifar-10-batches-py/test_batch'
        self.LOAD_CIFAR_DATA( imgFileName )

        self.Show()

    def LOAD_CIFAR_DATA(self, filename ):
        with open(filename,'rb') as f:
            dataset = pickle.load( f )

            # print len(dataset['filenames'])
            X = dataset['data']
            Y = dataset['labels']

            self.X_input_list.append( np.reshape( X, ( 10000, 3, 32, 32 ) ) )
            self.Y_input_list.append( Y )


    def LOAD_CIFAR_LABELS(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load( f )
            print obj
            self.label_names =  obj['label_names']

    def Show(self):  # change the shape of the input to be 50000 * 3072 with three channels of one pixel being together
        num_in_batch = self.X_input_list[0].shape[0]

        for i in range( len( self.X_input_list ) ):
            for j in range( num_in_batch ):
                img = self.X_input_list[i][j]
                img_merged = cv2.merge([img[0], img[1], img[2]])

                self.X_input[ i * num_in_batch + j ] = np.reshape( img_merged, ( 32*32*3 ) )
                # print self.Y_input_list[ i * num_in_batch + j ]
                self.Y_input[ i * num_in_batch + j ] = self.Y_input_list[ i ][ j ]
                # cv2.imshow("Image", img_merged)
                # print self.label_names[self.Y_input[i*num_in_batch + j]]
                # cv2.waitKey(0)
        # self.X_input = np.array( X_input_tmp )
        # self.Y_input = self.tmp_Y_input
        #
        self.test_X = self.X_input[ 50000:,: ]
        self.test_Y = self.Y_input[50000:]


        self.X_input = self.X_input[ 0: 50000, : ]
        self.Y_input = self.Y_input[ 0: 50000]

        # print self.X_input.shape
        # print self.Y_input.shape
        #
        # print self.test_X.shape
        # print self.test_Y.shape

def main():
    knn = KNN()
    knn.predict()





if __name__ == '__main__':
    main()
