import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import mahotas as mt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from skimage import feature
import time

def brisk(img):
    brisk = cv.BRISK_create()
    (kps, descs) = brisk.detectAndCompute(img, None)
    descs = descs.mean(axis=0)   
    return descs

def akaze(img):
    akaze = cv.AKAZE_create()
    (kps, descs) = akaze.detectAndCompute(img, None)
    descs = descs.mean(axis=0)
    return descs

def kaze(img):
    kaze = cv.KAZE_create()
    (kps, descs) = kaze.detectAndCompute(img, None)
    descs = descs.mean(axis=0)    
    return descs

def orb(img):
    orb = cv.ORB_create()
    (kps, descs) = orb.detectAndCompute(img, None)
    descs = descs.mean(axis=0)
    return descs

def hist(img):
    hist = cv.calcHist([img], [0], None, [512], [0, 512])
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-9)
    return hist.flatten()

def haralick(img):
    textures = mt.features.haralick(img)
    ht_mean = textures.mean(axis=0)   
    return ht_mean
    
def choose_alg(alg):
    if alg == "1":
        alg = kaze
    elif alg == "2":
        alg = akaze
    elif alg == "3":
        alg = brisk
    elif alg == "4":
        alg = orb
    elif alg == "5":
        alg = hist
    elif alg == "6":
        alg = haralick
    return alg

def main():
    names = ['circinatum', 'garryana', 'glabrum', 'kelloggii', 'macrophyllum', 'negundo'] 
    path = "isolated\\"
    files = []
    imgs = []
    labels = []
    alg = 0
    sys.setrecursionlimit(10**4)

    print("[START]")
    print("Choose the algorithm:\n  1 = KAZE\n  2 = AKAZE\n  3 = BRISK\n  4 = ORB\n  5 = Histogram\n  6 = Haralick")
    alg = input()
    alg = choose_alg(alg)
   
    print("[SEARCHING FOR IMAGES]")
    start = time.time()
    for r, d , f in os.walk(path):
        for file in f:
            files.append(os.path.join(r,file))
    end = time.time()
    print("    DONE - " + str(end-start) + "s\n")

    print("[EXTRACTING FEATURES FROM IMAGES]")
    start = time.time()    
    for file in files:
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = alg(img)
        imgs.append(img)
        for name in names:
            if name in file:
                labels.append(name)
                break
    end = time.time()
    print("    DONE - " + str(end-start) + "s\n")

    print("[TESTING]")
    start = time.time()
    scl = StandardScaler()
    imgs = scl.fit_transform(imgs)

    xTrain, xTest, yTrain, yTest = train_test_split(imgs, labels, test_size=0.2, random_state=0)

	# Support Vector Classification test
    svc = LinearSVC(max_iter=9000)
    svc.fit(xTrain, yTrain)
	
	# Cross validation score test (sklearn)
    score = cross_val_score(svc, imgs, labels, cv=10)
    end = time.time()
    print("    DONE - " + str(end-start) + "s\n")

    print("==========[RESULTS]=========\n")
    print("[CROSS VALIDATION SCORE]")
    print(score)
    print("Mean of all iterations: " + str(np.mean(score)) + "\n")

    print("[LINEAR SUPPORT VECTOR CLASSIFICATION SCORE]")
    score = svc.score(xTrain, yTrain)
    print('Training model accuracy:  ' + str(score))
    score = svc.score(xTest, yTest)
    print('Testing model accuracy :  ' + str(score) + "\n")

    print("[CLASSIFICATION REPORT]")
    test_pred = svc.predict(xTest)   
    print(classification_report(test_pred, yTest, target_names=names))

if __name__=="__main__":
    main()
