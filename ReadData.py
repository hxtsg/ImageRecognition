import pickle
import cv2
import numpy as np
import random
class ReadData():
    def __init__(self):
        self.label_names = None
        self.X = None
        self.Y = None

    def LOAD_CIFAR_DATA(self, filename):
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
            print dataset.keys()
            # print len(dataset['filenames'])
            X = dataset['data']
            Y = dataset['labels']
            self.X = np.reshape(X, (10000, 3, 32, 32))
            self.Y = np.array(Y)

    def LOAD_CIFAR_LABELS(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            print obj
            self.label_names = obj['label_names']

    def Read(self, imgFileName, labelFileName):
        self.LOAD_CIFAR_LABELS(filename=labelFileName)
        self.LOAD_CIFAR_DATA(filename=imgFileName)

    def Show(self):
        example_nums = self.X.shape[0]
        for i in range(example_nums):
            img = self.X[i]
            img_merged = cv2.merge([img[0], img[1], img[2]])
            cv2.imshow("Image", img_merged)
            print self.label_names[self.Y[i]]
            cv2.waitKey(0)


def main():
    readData = ReadData()
    imgFileName = './cifar-10-batches-py/data_batch_2'
    labelFileName = './cifar-10-batches-py/batches.meta'
    readData.Read(imgFileName, labelFileName)
    readData.Show()


if __name__ == '__main__':
    # main()
    arr = [ 5, 6, 7, 1, 2 ]
    for i in range(20):
        print arr[random.randint( 0, len( arr ) - 1 )]