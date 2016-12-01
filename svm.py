import pickle
import random
import numpy as np
from PIL import Image
import cv2




class SVM():
    def __init__(self):
        self.data = Data()
        self.data.Run()
        self.BATCH_SIZE = 256
        self.weights = np.random.rand(10,3073)     # a matrix of 10 * 3073
        self.stepSize = 0.00001
        self.delta = 10
        self.lamb = 0.001



    def train(self):
        train_times = 10000
        for i in range( train_times ):

            self.weights += -self.stepSize * self.calc_grad()
            print i, " ", self.calc_loss()

    def predict(self):
        predict_y = np.zeros( self.data.test_Y.shape[0] )
        example_nums = self.data.test_X.shape[0]
        for i in range( example_nums ):
            X = np.concatenate( ( self.data.test_X[ i ], np.array( [ 1 ] ) ) )
            scores = np.matmul( self.weights, X )
            predict_y[ i ] = np.argmax( scores )
        print "accuracy:", np.mean( predict_y == self.data.test_Y )


    def calc_grad(self):
        # get batch
        data_batch = np.array(random.sample( self.data.All_data, self.BATCH_SIZE ))

        example_X = data_batch[:,:-1]
        example_Y = data_batch[:,-1:]

        example_num = example_X.shape[0]
        class_num = 10
        grad_weight = np.zeros( ( 10, 3073 ) )
        tmp_grad_weight = np.zeros( ( 10, 3073 ) )


        for i in range( example_num ):
            X = np.concatenate(( example_X[ i ], np.array( [ 1 ] ))) # 3073 * 1
            indicator = np.matmul( self.weights, X )   # 10 * 1

            grad_miss_cnt = 0

            for j in range(class_num):
                if indicator[ j ] - indicator[ example_Y[ i ][ 0 ] ] + self.delta > 0:
                    tmp_grad_weight[ j ] = X
                    grad_miss_cnt += 1
                else:
                    tmp_grad_weight[ j ] = np.zeros( ( 1 * 3073 ) )
            tmp_grad_weight[ example_Y[ i ][ 0 ] ] = -1 * grad_miss_cnt * X

            grad_weight += tmp_grad_weight

        grad_weight = grad_weight / example_num
        return grad_weight

    def calc_loss( self ):  # X is the dimension of 1 * 3073

        # loss term



        loss_term = 0.0
        example_nums = self.data.X_input.shape[0]
        for i in range( example_nums ):
            bias_term = np.array( [1] )

            X = np.concatenate( ( self.data.X_input[ i ], bias_term ) )
            scores = np.matmul( self.weights, X )

            for j in range( scores.shape[0] ):
                if j != self.data.Y_input[ i ]:
                    loss_term += max( 0, scores[ j ] - scores[ self.data.Y_input[ i ] ] + self.delta )

        loss_term = loss_term / example_nums



        # regularization term

        reg_term = self.lamb * np.sum( np.sum( np.square( self.weights ) ) )


        return loss_term




class Data():
    def __init__(self):
        self.label_names = None


        self.X_input_list = []
        self.Y_input_list = []


        self.X_input = np.zeros( (60000, 3 * 32 * 32) )   # training data and labels
        self.Y_input = np.zeros( (60000), dtype=np.int32)

        self.test_X = None   # test data and labels
        self.test_Y = None

        self.All_data = None

    def Run(self):

        labelFileName = './cifar-10-batches-py/batches.meta'
        self.LOAD_CIFAR_LABELS(filename=labelFileName)

        for i in range(1,6):
            imgFileName = './cifar-10-batches-py/data_batch_' + str( i )
            self.LOAD_CIFAR_DATA(imgFileName )

        imgFileName = './cifar-10-batches-py/test_batch'
        self.LOAD_CIFAR_DATA( imgFileName )

        self.Show()
        self.Patch_Together()
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

    def Patch_Together(self):  # push X_input and Y_input into one matrix of 50000 * ( 3072 + 1 ) with 1 means the y label

        self.All_data = np.concatenate( ( self.X_input, np.reshape( self.Y_input, ( self.Y_input.shape[ 0 ], 1 )) ), axis= 1 )
        return None
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


if __name__ == '__main__':
    svm = SVM()
    svm.train()
    svm.predict()
    # a = np.array( [ [ 1, 2 ], [ 3, 4 ]  ] )
    # b = np.array( [ [ 5, 6 ], [ 7, 8 ]  ] )
    # c = np.concatenate( ( a,b ), axis= 1 )
    # print c