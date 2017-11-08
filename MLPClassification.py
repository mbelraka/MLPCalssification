import os
import struct
import array
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from matplotlib import pyplot as plot
from numba import jit

#reading

@jit(cache=True)
def read_dataset(set = 1, n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

    path = os.getcwd()

    if(set == 1):

        image_filepath = os.path.join(path, "datasets/train-images-idx3-ubyte")
        label_filepath = os.path.join(path, "datasets/train-labels-idx1-ubyte")
    else:
        image_filepath = os.path.join(path, "datasets/t10k-images-idx3-ubyte")
        label_filepath = os.path.join(path, "datasets/t10k-labels-idx1-ubyte")

    image_file = open(image_filepath, 'rb')
    magic_nr, size, rows, columns = struct.unpack(">IIII", image_file.read(16))
    image = array.array("B", image_file.read())
    image_file.close()

    label_file = open(label_filepath, 'rb')
    magic_nr, size = struct.unpack(">II", label_file.read(8))
    label = array.array("b", label_file.read())
    label_file.close()
    l = len(label)

    images = np.zeros((l, 784), dtype=np.uint8)
    labels = np.zeros((l, 1), dtype=np.int8).flatten()

    for i in range(l):

        images[i] = np.array(image[i * rows * columns : (i + 1) * rows * columns])
        labels[i] = label[i]

    return images, labels

images_train, labels_train = read_dataset() #try to read training dataset

images_test, labels_test = read_dataset(2) #try to read test dataset

#MLP Classification
@jit(cache=True)
def mlp():

    global images_train
    global labels_train
    global images_test
    global labels_test

    mlp = MLPClassifier(hidden_layer_sizes =(340, 340), random_state=0, max_iter=100000)

    mlp.fit(images_train[:20000], labels_train[:20000])

    score = mlp.score(images_test[:5000], labels_test[:5000])

    mlp.early_stopping = True
    mlp_training_size, mlp_training_score, mlp_test_score = learning_curve(mlp, images_train[:2000], labels_train[:2000], n_jobs= 8)

    mlp_training_score_mean = np.mean(mlp_training_score, axis=1)
    mlp_training_score_std = np.mean(mlp_training_score, axis=1)

    mlp_test_score_mean = np.mean(mlp_test_score, axis=1)
    mlp_test_score_std = np.mean(mlp_test_score, axis=1)

    score *= 100

    print("MLPClassification score =", score, "% \n")

    return mlp, mlp_training_size, mlp_training_score, mlp_test_score, mlp_training_score_mean, mlp_training_score_std, mlp_test_score_mean, mlp_test_score_std


#SVM Classification
#try to run different times
@jit(cache=True)
def fSVC():

    global images_train
    global labels_train
    global images_test
    global labels_test

    svc = SVC(C=3, kernel='poly', random_state=0, verbose=True)
    svc.fit(images_train[:20000], labels_train[:20000])
    score = svc.score(images_test[:5000], labels_test[:5000])
    #svc_scores = cross_val_score(svc, images_train[:2000], labels_train[:2000], cv=5)
    svc_training_size, svc_training_score, svc_test_score = learning_curve(svc, images_train[:20000], labels_train[:20000], n_jobs=8)

    svc_training_score_mean = np.mean(svc_training_score, axis=1)
    svc_training_score_std = np.mean(svc_training_score, axis=1)

    svc_test_score_mean = np.mean(svc_test_score, axis=1)
    svc_test_score_std = np.mean(svc_test_score, axis=1)

    score *= 100
    print("\n\nSVMClassification score =", score, "%")

    return svc, svc_training_size, svc_training_score, svc_test_score, svc_training_score_mean, svc_training_score_std, svc_test_score_mean, svc_test_score_std


mlp, mlp_training_size, mlp_training_score, mlp_test_score, mlp_training_score_mean, mlp_training_score_std, mlp_test_score_mean, mlp_test_score_std = mlp()

svc, svc_training_size, svc_training_score, svc_test_score, svc_training_score_mean, svc_training_score_std, svc_test_score_mean, svc_test_score_std = fSVC()

plot.figure()
plot.title("learning curves - MLPClassification")
plot.xlabel("Training instances")
plot.ylabel("score")
plot.grid()
plot.fill_between(mlp_training_size, mlp_training_score_mean - mlp_training_score_std, mlp_training_score_mean + mlp_training_score_std, alpha=0.1)
plot.plot(mlp_training_size, mlp_training_score, 'o-', color="b", label = "Training score")
plot.fill_between(mlp_training_size, mlp_test_score_mean - mlp_test_score_std, mlp_test_score_mean + mlp_test_score_std, alpha=0.1)
plot.plot(mlp_training_size, mlp_test_score, 'o-', color="g", label = "Validation score")
plot.legend(loc="best")
plot.show()


plot.figure()
plot.title("learning curves - SVMClassification")
plot.xlabel("Training instances")
plot.ylabel("score")
plot.grid()
plot.fill_between(svc_training_size, svc_training_score_mean - svc_training_score_std, svc_training_score_mean + svc_training_score_std, alpha=0.1)
plot.plot(svc_training_size, svc_training_score, 'o-', color="b", label = "Training score")
plot.fill_between(svc_training_size, svc_test_score_mean - svc_test_score_std, svc_test_score_mean + svc_test_score_std, alpha=0.1)
plot.plot(svc_training_size, svc_test_score, 'o-', color="g", label = "Validation score")
plot.legend(loc="best")
plot.show()

#PCA

pca = PCA()

images_train_proj = pca.fit_transform(images_train[:20000], labels_train[:20000])
images_test_proj = pca.fit_transform(images_test[:5000], labels_test[:5000])

mlp.early_stopping = False

mlp.fit(images_train_proj, labels_train[:20000])
score_pca = mlp.score(images_test_proj, labels_test[:5000])

score_pca *= 100

print("score after applying PCA", score_pca, "%")


