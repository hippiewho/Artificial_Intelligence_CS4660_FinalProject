import cv2
import glob
import os
import numpy as np
import operator


def load_images(path, outfile, width=100, height=100):

    print('Processing images...', path)

    # check if file exist
    if not os.path.exists(outfile):
        with open(outfile, 'wb') as f:

            img_canvas = np.zeros((height, width), dtype=np.uint8)
            # https://pymotw.com/2/glob/
            for filename in glob.glob(path + '/*.jpg'):

                im = cv2.imread(filename)
                imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                imgray = cv2.bitwise_not(imgray)

                img_canvas.fill(0)
                img_canvas = imgray[:height, :width]

                f.write(img_canvas.flatten())

                print(filename, '...', 'done')

    # load file if present
    data = np.fromfile(outfile, dtype=np.uint8)
    data.shape = (-1, width * height)

    return data


def extract_features(images):
    return images


def euclideanDistance(instance1, instance2):

    instance1, instance2 = np.asarray(instance1), np.asarray(instance2)
    eucl_dist = np.linalg.norm(instance1 - instance2, 2, 0)

    return eucl_dist


def getneighbors(train_X, train_Y, testInsance, k):
    neighbors = []
    for i in range(len(train_X)):
        distance = euclideanDistance(train_X[i], testInsance)
        neighbors.append((train_X[i], train_Y[i], distance))

    neighbors.sort(key=operator.itemgetter(2))

    return neighbors[:k]  # indices of k-nearest neighbors in training data


def assignLabel(neighbors):
    assignedclass = {}
    for neighbor in neighbors:
        label = neighbor[1]
        assignedclass[label] = assignedclass.get(label, 0) + 1

    # sort classes on the basis of votes it has recieved
    assignedclass = sorted(assignedclass.items(),
                           key=operator.itemgetter(1), reverse=True)

    return assignedclass[0][0]  # return most voted class


def getAccuracy(true_labels, predictions):
    correct_predictions = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predictions[i]:
            correct_predictions += 1

    return float(correct_predictions) / len(true_labels)

def shuffle_array(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
    # https://discuss.analyticsvidhya.com/t/how-to-choose-the-value-of-k-in-knn-algorithm/2606
    K = 5
    # K = 1
    TRAIN_SIZE = 150
    # Change the file folders here
    # Create an output file to output the binary data resulting from running
    # the KNN algorithm      
    #                        File images here, name for binary file here
    #                         You will need to add the binary file name
    #                         I have a few binary files already in the main program folder
    CLASS_A = load_images('./normal', 'data6NORMAL6.bin')
    CLASS_B = load_images('./diseasedPOSASTIG', 'data7NORMAL7v2.bin')
    CLASS_C = load_images('./normalASTIG', 'data6aNORMAL6av2.bin')

    CLASS_A_X = extract_features(CLASS_A)
    CLASS_B_X = extract_features(CLASS_B)
    CLASS_C_X = extract_features(CLASS_C)

    X = np.concatenate((CLASS_A_X, CLASS_B_X, CLASS_C_X), axis=0)
    Y = np.asarray([0] * len(CLASS_A) + [1] * len(CLASS_B) + [2] * len(CLASS_C)+ [3])
    
    X, Y = shuffle_array(X, Y)

    train_X, test_X = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
    train_Y, test_Y = Y[:TRAIN_SIZE], Y[TRAIN_SIZE:]
    # There will be some noticeable lag as you run the algorithm
    print('Applying K Nearest Neighbors, please stand-by...')
    # https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7
    # good explanation to KNN and some applications for it
    predictions = []
    for i in range(len(test_X)):
        predicted_label = assignLabel(getneighbors(train_X, train_Y, test_X[i], K))
        predictions.append(predicted_label)

    print('Accuracy: {0:.8%}'.format(getAccuracy(test_Y, predictions)))
    # sources: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
    # sources: https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
    # https://www.dataquest.io/blog/k-nearest-neighbors-in-python/
    # https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/

