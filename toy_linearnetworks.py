import numpy as np
import matplotlib.pyplot as plt
import math


class Data():
    def __init__(self):
        self.N = 100  # number of points per class
        self.D = 2  # dimensionality
        self.K = 3  # number of classes
        self.X = np.zeros((self.N * self.K, self.D))  # data matrix (each row = single example)
        self.y = np.zeros(self.N * self.K, dtype='uint8')  # class labels
        self.W = 0.01 * np.random.randn(self.D, self.K )
        self.b = 0.01 * np.random.randn(1, self.K)

        self.loss = np.zeros( ( self.X.shape[ 0 ] ) )
        self.probs = np.zeros( ( self.N * self.K, self.K ) )
        self.scores = None
        self.reg = 1e-3
        self.step_size = 1e-0
        for j in xrange(self.K):
            ix = range(self.N * j, self.N * (j + 1))
            r = np.linspace(0.0, 1, self.N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, self.N) + np.random.randn(self.N) * 0.2  # theta
            self.X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            self.y[ix] = j
        # lets visualize the data:
    def plotting(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    def calc_loss(self):
        self.scores = np.dot( self.X, self.W ) + self.b
        self.scores_exp = np.exp( self.scores )
        example_num = self.X.shape[0]
        for i in range( example_num ):
            # print self.probs[i].shape
            # print self.scores_exp.shape
            # print np.sum( self.scores_exp[ i ] )
            self.probs[ i ] = self.scores_exp[ i ] / np.sum( self.scores_exp[ i ] )
            self.loss[ i ] = -1 * np.log( self.probs[ i ][ self.y[ i ] ] )
        reg_term = 0.5 * self.reg * np.sum(np.sum(self.W * self.W))
        return np.sum( self.loss ) / example_num + reg_term

    def train(self):
        example_num = self.X.shape[0]

        dscore = self.probs
        for i in range( example_num ):
            dscore[ i ][ self.y[ i ] ] -= 1
        dscore /= example_num
        dW = np.dot(self.X.T,dscore) + self.reg * self.W
        db = np.sum( dscore, axis= 0, keepdims= True )

        self.W -= dW * self.step_size
        self.b -= db * self.step_size

    def train_all(self):

        train_times = 200
        for i in range( train_times ):
            print 'The current loss is: ',self.calc_loss()
            self.train()

    def predict(self):
        self.scores = np.dot( self.X, self.W ) + self.b
        predict_y = np.argmax( self.scores, axis= 1 )

        print 'accuracy is: ', np.mean( predict_y == self.y )


if __name__ == "__main__":
    data = Data()
    data.plotting()
    data.train_all()
    data.predict()