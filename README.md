
Warsaw, 21 stof January 2020

## Mikołaj Rajch

# Plant Species Recognition using Feature Vectors

## Introduction to Image Processing and Computer Vision

## Project 2


## Contents

- 1 Introduction
   - 1.1 Project goals description
   - 1.2 Program description
- 2 Feature Extraction
   - 2.1 KAZE features
   - 2.2 AKAZE features
   - 2.3 BRISK features
   - 2.4 ORB features
   - 2.5 Histogram features
   - 2.6 Haralick features
- 3 Testing
- 4 Results
   - 4.1 Summary
   - 4.2 Detailed test scores (program output)
- 5 References


## 1 Introduction

### 1.1 Project goals description

The goal of this project is to make a program that recognizes the plant’s species
based on its leaves shape usingfeature extraction.
In machine learning, pattern recognition and in image processing, feature extrac-
tion starts from an initial set of measured data and builds derived values (features)
intended to be informative and non-redundant, facilitating the subsequent learning
and generalization steps, and in some cases leading to better human interpretations.[1]
For this project, a data set containing multiple photos of 6 different plant species’
leaves. Every photo was done in the same style. Leaves themselves were similar to
each other (of course within their respective species) in shape, however they differed
in rotation.

### 1.2 Program description

In this project, few different algorithms for extracting feature vectors from images
were used. Each of them will be separately shortly described.

First step of the program is simple choosing the algorithm and reading all images’
paths from all species into an array:

```
print ( " [START] " )
print ( "Choose the algorithm :\ n 1 = KAZE\n 2 = AKAZE\n 3 =
BRISK\n 4 = ORB\n 5 = Histogram\n 6 = Haralick " )
alg = input ()
alg = choose_alg ( alg )

print ( " [SEARCHING FOR IMAGES] " )
s t a r t = time. time ()
f o r r , d , f in os. walk ( path ) :
f o r f i l e in f :
f i l e s. append ( os. path. j o i n ( r , f i l e ) )
end = time. time ()
print ( " DONE− " + s t r ( end−s t a r t ) + " s \n" )
```
Then, for every path in the array, we read its representative picture. Its features
are then extracted using previously chosen algorithm. Also a corresponding species
label is given to every picture for testing purposes:

```
print ( " [EXTRACTING FEATURES FROM IMAGES] " )
s t a r t = time. time ()
f o r f i l e in f i l e s :
img = cv. imread ( f i l e )
img = cv. cvtColor (img , cv .COLOR_BGR2GRAY)
img = alg ( img )
imgs. append ( img )
f o r name in names :
i f name in f i l e :
l a b e l s. append (name)
break
end = time. time ()
print ( " DONE− " + s t r ( end−s t a r t ) + " s \n" )
```
This part of the code is pretty straight forward. Now I will briefly describe every
feature extracting algorithm used in the program.


## 2 Feature Extraction

### 2.1 KAZE features

KAZE Features is a novel 2D feature detection and description method that oper-
ates completely in a nonlinear scale space. Previous methods such as SIFT or SURF
find features in the Gaussian scale space (particular instance of linear diffusion).
However, Gaussian blurring does not respect the natural boundaries of objects and
smoothes in the same degree details and noise when evolving the original image
through the scale space.
By means of nonlinear diffusion we can detect and describe features in nonlinear
scale spaces keeping important image details and removing noise as long as we evolve
the image in the scale space. We use variable conductance diffusion which is one of
the simplest cases of nonlinear diffusion. The nonlinear scale space is build efficiently
by means of Additive Operator Splitting (AOS) schemes, which are stable for any
step size and are parallelizable.[2]

```
def kaze ( img ) :
kaze = cv. KAZE_create ()
( kps , descs ) = kaze. detectAndCompute (img , None)
descs = descs. mean( axis =0)
return descs
```
### 2.2 AKAZE features

Accelerated KAZE Features uses a novel mathematical framework called Fast Ex-
plicit Diffusion (FED) embedded in a pyramidal framework to speed-up dramatically
the nonlinear scale space computation. In addition, we compute a robust Modified-
Local Difference Binary (M-LDB) descriptor that exploits gradient information from
the nonlinear scale space. AKAZE obtains comparable results to KAZE in some
datasets, while being several orders of magnitude faster.[2]

```
def akaze ( img ) :
akaze = cv. AKAZE_create ()
( kps , descs ) = akaze. detectAndCompute (img , None)
descs = descs. mean( axis =0)
return descs
```

### 2.3 BRISK features

Binary Robust Invariant Scalable Keypoints (BRISK) detects corners using AGAST
algorithm and filters them with FAST Corner score while searching for maxima in
the scale space pyramid. BRISK description is based on identifying the characteristic
direction of each feature for achieving rotation invariance. To cater illumination in-
variance results of simple brightness tests are also concatenated and the descriptor is
constructed as a binary string. BRISK features are invariant to scale, rotation, and
limited affine changes. [3]

```
def brisk ( img ) :
brisk = cv. BRISK_create ()
( kps , descs ) = brisk. detectAndCompute (img , None)
descs = descs. mean( axis =0)
return descs
```
### 2.4 ORB features

Oriented FAST and Rotated BRIEF (ORB) algorithm is a blend of modified FAST
(Features from Accelerated Segment Test) detection and direction-normalized BRIEF
(Binary Robust Independent Elementary Features) description methods. FAST cor-
ners are detected in each layer of the scale pyramid and cornerness of detected points
is evaluated using Harris Corner score to filter out top quality points. As BRIEF de-
scription method is highly unstable with rotation, thus a modified version of BRIEF
descriptor has been employed. ORB features are invariant to scale, rotation and lim-
ited affine changes.[3]

```
def orb ( img ) :
orb = cv. ORB_create ()
( kps , descs ) = orb. detectAndCompute (img , None)
descs = descs. mean( axis =0)
return descs
```
### 2.5 Histogram features

In this method we calculate the images histogram using OpenCV function calcHist().
We do this to analyze local representation of textures in the image (extracted by com-
paring each pixel with its surrounding neighborhood of pixels). After normalization
of the provided histogram we return the desired feature vector.

```
def h i s t ( img ) :
h i s t = cv. calcHist ( [ img ] , [ 0 ] , None , [ 5 1 2 ] , [0 , 512])
h i s t = h i s t. astype ( " f l o a t " )
h i s t /= ( h i s t. sum() + 1e−9)
return h i s t. f l a t t e n ()
```

### 2.6 Haralick features

Haralick features are texture features, based on the adjacency matrix. The adja-
cency matrix stores in position (i,j) the number of times that a pixel takes the value
i next to a pixel with the value j. Given different ways to define next to, you obtain
slightly different variations of the features. Standard practice is to average them out
across the directions to get some rotational invariance. They can be computed for
2-D or 3-D images.[4]

```
def haralick ( img ) :
textures = mt. f e a t u r e s. haralick ( img )
ht_mean = textures. mean( axis =0)
return ht_mean
```
## 3 Testing

Every feature extracting algorithm has been tested in terms of its accuracy. Their
respective computation times were also measured. For testing purposes the features
were standardized by removing the mean and scaling to unit variance using Stan-
dardSclaer(). For accuracy measurement three different testing methods have been
used:

- cross_val_score() function provided by sklearn,
- score() function built-in the SVM used,
- predict() function, also built-in the SVM used.
The first method uses cross-validation - it starts by shuffling the data (to prevent
any unintentional ordering errors) and splitting it intokfolds. Then k models are fit
on k−k^1 of the data (called the training split) and evaluated on^1 k of the data (called
the test split). The results from each evaluation are averaged together for a final
score, then the final model is fit on the entire dataset for operationalization. In this
particular case a 10-fold cross validation have been used.
The next two functions use training and testing data sets to compute the re-
sults. The arrays with feature vectors and labels have been split into two sets: one
for training, and the other one for the actual test. The split has been done using
train_test_split() function provided by sklearn. The proportion of number of ele-
ments in sets was 80/20.
In this project, linear vector classification was used instead of regular vector clas-
sification, in the form of LinearSVC(). It is a type of SVM. In machine learning,
support-vector machines are supervised learning models with associated learning al-
gorithms that analyze data used for classification and regression analysis. Given a set
of training examples, each marked as belonging to one or the other of two categories,
an SVM training algorithm builds a model that assigns new examples to one category
or the other, making it a non-probabilistic binary linear classifier. The second test is
a mean accuracy score from both sets (training and testing).


Finally, the last test is done using function predict(), that simply predicts the
accuracy of class labels for samples in the testing set. The scores are presented in the
form of a table, showing classification metrics of the experiment.

```
print ( " [TESTING] " )
s t a r t = time. time ()
s c l = StandardScaler ()
imgs = s c l. fit_transform ( imgs )

xTrain , xTest , yTrain , yTest = train_test_split ( imgs , labels ,
test_size =0.2 , random_state=0)

# Support Vector C l a s s i f i c a t i o n t e s t
svc = LinearSVC ( max_iter=9000)
svc. f i t ( xTrain , yTrain )

# Cross validation score t e s t ( sklearn )
score = cross_val_score ( svc , imgs , labels , cv=10)
end = time. time ()
print ( " DONE− " + s t r ( end−s t a r t ) + " s \n" )

print ( "==========[RESULTS]=========\n" )
print ( " [CROSS VALIDATION SCORE] " )
print ( score )
print ( "Mean of a l l i t e r a t i o n s : " + s t r (np. mean( score ) ) + "\n" )

print ( " [LINEAR SUPPORT VECTOR CLASSIFICATION SCORE] " )
score = svc. score ( xTrain , yTrain )
print ( ’ Training model accuracy : ’ + s t r ( score ) )
score = svc. score ( xTest , yTest )
print ( ’ Testing model accuracy : ’ + s t r ( score ) + "\n" )

print ( " [CLASSIFICATION REPORT] " )
test_pred = svc. predict ( xTest )
print ( c l a s s i f i c a t i o n _ r e p o r t ( test_pred , yTest , target_names=
names ) )
```

## 4 Results

### 4.1 Summary

The main coefficient in the testing stage was accuracy, but time of computation
for all stages of the program (with the algorithms processing all the images being the
most important stage) was also measured.
The basic scores summary is presented in the following table:

```
Algorithm Test accuracy (in %) Time (in s)
KAZE   98. 87640449438202    261. 27
AKAZE    95. 50561797752809    47. 08
BRISK    97. 75280898876404    145. 50
ORB    93. 25842696629213    8. 66
Histogram    85. 39325842696629    3. 83
Haralick    85. 39325842696629    34. 20
Table with mean accuracy and computation time for every algorithm
```
As we can see, the most precise algorithm for feature extraction turned out to be
the KAZE algorithm with nearly99%accuracy. For all subsets but one its precision
was a100%, with the exception being the garryana subset with score of94%, and the
mean of 10-fold cross validation test was over95%. Unfortunately, its computation
time was also the largest, being a little over 4 minutes, which makes it probably the
best algorithm for smaller data sets.
The AKAZE algorithm worked as expected - it’s precision is similar to that of the
KAZE algorithm (95%, with mean of 10-fold being92%), with much faster time - 47
seconds.
BRISK was overall probably the best algorithm used in this program - with very
high accuracy of almost98%, and with computation time almost 2 times faster than
that of KAZE. Mean of the cross validation test was94%. Precision prediction for
all but two subsets was a100%. What is interesting is that the subset not evaluated
perfectly in KAZE had precision of100%for BRISK. Overall, those two algorithms
performed very similarly.
ORB, as the last of shape features extracting algorithm, finished in only 8s. The
precision however was noticeably worse, with even training set accuracy not being
100%. The mean for cross validation test was 91%, and testing model accuracy -
93%.
Finally, the two last algorithms (one using histogram and haralick) performed
significantly worse. Algorithm based on histograms ended in just 3s, making it the
fastest, but its precision was only85%(cross-val being82%). Then, for Haralick the
precision for testing set was similar, but this was the only case with training model
having such a low score of only87%. Its time also was not the best, being 34s.
Overall, algorithms extracting shape features performed the best in terms of ac-
curacy and precision. Local textures algorithm (one based on histograms) was the
fastest, but also less precise. Texture feature extracting algorithm performed the
worst. More detailed test results are given below (whole program output).


### 4.2 Detailed test scores (program output)

#### ============================== KAZE =============================

#### [START]

Choose the algorithm:
1 = KAZE
2 = AKAZE
3 = BRISK
4 = ORB
5 = Histogram
6 = Haralick
1
[SEARCHING FOR IMAGES]
DONE - 0.007003068923950195s

#### [EXTRACTING FEATURES FROM IMAGES]

```
DONE - 261.27635312080383s
```
#### [TESTING]

```
DONE - 0.9364275932312012s
```
#### ==========[RESULTS]=========

#### [CROSS VALIDATION SCORE]

#### [0.97777778 0.95555556 0.93181818 0.93181818 1. 0.

#### 0.97727273 0.88636364 0.97727273 0.93181818]

Mean of all iterations: 0.

[LINEAR SUPPORT VECTOR CLASSIFICATION SCORE]
Training model accuracy: 1.
Testing model accuracy : 0.

#### [CLASSIFICATION REPORT]

```
precision recall f1-score support
circinatum 1.00 1.00 1.00 14
garryana 0.94 1.00 0.97 16
glabrum 1.00 1.00 1.00 17
kelloggii 1.00 1.00 1.00 17
macrophyllum 1.00 0.95 0.97 19
negundo 1.00 1.00 1.00 6

accuracy 0.99 89
macro avg 0.99 0.99 0.99 89
weighted avg 0.99 0.99 0.99 89
```

#### >>>

#### ============================== AKAZE =============================

#### [START]

Choose the algorithm:
1 = KAZE
2 = AKAZE
3 = BRISK
4 = ORB
5 = Histogram
6 = Haralick
2
[SEARCHING FOR IMAGES]
DONE - 0.006997585296630859s

#### [EXTRACTING FEATURES FROM IMAGES]

```
DONE - 47.084415912628174s
```
#### [TESTING]

```
DONE - 0.9666883945465088s
```
#### ==========[RESULTS]=========

#### [CROSS VALIDATION SCORE]

#### [0.86666667 0.95555556 0.93181818 0.93181818 0.90909091 0.

#### 0.97727273 0.84090909 0.95454545 0.93181818]

Mean of all iterations: 0.

#### [LINEAR SUPPORT VECTOR CLASSIFICATION SCORE]

Training model accuracy: 1.
Testing model accuracy : 0.

#### [CLASSIFICATION REPORT]

```
precision recall f1-score support
circinatum 0.93 1.00 0.96 13
garryana 1.00 0.94 0.97 18
glabrum 1.00 0.94 0.97 18
kelloggii 1.00 0.94 0.97 18
macrophyllum 0.89 1.00 0.94 16
negundo 0.83 0.83 0.83 6

accuracy 0.96 89
macro avg 0.94 0.94 0.94 89
weighted avg 0.96 0.96 0.96 89
```

#### >>>

#### ============================== BRISK =============================

#### [START]

Choose the algorithm:
1 = KAZE
2 = AKAZE
3 = BRISK
4 = ORB
5 = Histogram
6 = Haralick
3
[SEARCHING FOR IMAGES]
DONE - 0.006003856658935547s

#### [EXTRACTING FEATURES FROM IMAGES]

```
DONE - 145.50773239135742s
```
#### [TESTING]

```
DONE - 0.8757138252258301s
```
#### ==========[RESULTS]=========

#### [CROSS VALIDATION SCORE]

#### [0.93333333 0.91111111 0.97727273 0.97727273 0.95454545 0.

#### 0.97727273 0.88636364 0.95454545 0.95454545]

Mean of all iterations: 0.

#### [LINEAR SUPPORT VECTOR CLASSIFICATION SCORE]

Training model accuracy: 1.
Testing model accuracy : 0.

#### [CLASSIFICATION REPORT]

```
precision recall f1-score support
circinatum 1.00 1.00 1.00 14
garryana 1.00 0.94 0.97 18
glabrum 0.94 1.00 0.97 16
kelloggii 1.00 1.00 1.00 17
macrophyllum 0.94 1.00 0.97 17
negundo 1.00 0.86 0.92 7

accuracy 0.98 89
macro avg 0.98 0.97 0.97 89
weighted avg 0.98 0.98 0.98 89
```

#### >>>

#### ============================== ORB =============================

#### [START]

Choose the algorithm:
1 = KAZE
2 = AKAZE
3 = BRISK
4 = ORB
5 = Histogram
6 = Haralick
4
[SEARCHING FOR IMAGES]
DONE - 0.0060079097747802734s

#### [EXTRACTING FEATURES FROM IMAGES]

```
DONE - 8.668512344360352s
```
#### [TESTING]

```
DONE - 0.6857805252075195s
```
#### ==========[RESULTS]=========

#### [CROSS VALIDATION SCORE]

#### [0.84444444 0.93333333 0.95454545 0.97727273 0.88636364 0.

#### 0.93181818 0.86363636 0.93181818 0.93181818]

Mean of all iterations: 0.

#### [LINEAR SUPPORT VECTOR CLASSIFICATION SCORE]

Training model accuracy: 0.
Testing model accuracy : 0.

#### [CLASSIFICATION REPORT]

```
precision recall f1-score support
circinatum 1.00 1.00 1.00 14
garryana 1.00 0.85 0.92 20
glabrum 0.94 0.89 0.91 18
kelloggii 1.00 1.00 1.00 17
macrophyllum 0.89 0.94 0.91 17
negundo 0.50 1.00 0.67 3

accuracy 0.93 89
macro avg 0.89 0.95 0.90 89
weighted avg 0.95 0.93 0.94 89
```

#### >>>

============================== Histogram =============================
[START]
Choose the algorithm:
1 = KAZE
2 = AKAZE
3 = BRISK
4 = ORB
5 = Histogram
6 = Haralick
5
[SEARCHING FOR IMAGES]
DONE - 0.006997346878051758s

#### [EXTRACTING FEATURES FROM IMAGES]

```
DONE - 3.839423418045044s
```
#### [TESTING]

```
DONE - 7.777447938919067s
```
#### ==========[RESULTS]=========

#### [CROSS VALIDATION SCORE]

#### [0.82222222 0.84444444 0.70454545 0.84090909 0.88636364 0.

#### 0.97727273 0.63636364 0.79545455 0.84090909]

Mean of all iterations: 0.

#### [LINEAR SUPPORT VECTOR CLASSIFICATION SCORE]

Training model accuracy: 0.
Testing model accuracy : 0.

#### [CLASSIFICATION REPORT]

```
precision recall f1-score support
circinatum 0.79 0.79 0.79 14
garryana 0.94 0.94 0.94 17
glabrum 0.88 0.88 0.88 17
kelloggii 0.88 0.79 0.83 19
macrophyllum 0.72 0.87 0.79 15
negundo 1.00 0.86 0.92 7

accuracy 0.85 89
macro avg 0.87 0.85 0.86 89
weighted avg 0.86 0.85 0.86 89
```

#### >>>

============================== Haralick =============================
[START]
Choose the algorithm:
1 = KAZE
2 = AKAZE
3 = BRISK
4 = ORB
5 = Histogram
6 = Haralick
6
[SEARCHING FOR IMAGES]
DONE - 0.006997585296630859s

#### [EXTRACTING FEATURES FROM IMAGES]

```
DONE - 34.20693063735962s
```
#### [TESTING]

```
DONE - 1.4216194152832031s
```
#### ==========[RESULTS]=========

#### [CROSS VALIDATION SCORE]

#### [0.86666667 0.82222222 0.88636364 0.84090909 0.90909091 0.

#### 0.79545455 0.72727273 0.77272727 0.88636364]

Mean of all iterations: 0.

#### [LINEAR SUPPORT VECTOR CLASSIFICATION SCORE]

Training model accuracy: 0.
Testing model accuracy : 0.

#### [CLASSIFICATION REPORT]

```
precision recall f1-score support
circinatum 0.71 0.71 0.71 14
garryana 0.71 1.00 0.83 12
glabrum 0.88 0.83 0.86 18
kelloggii 1.00 0.77 0.87 22
macrophyllum 0.94 1.00 0.97 17
negundo 0.83 0.83 0.83 6

accuracy 0.85 89
macro avg 0.85 0.86 0.85 89
weighted avg 0.87 0.85 0.85 89
```

## 5 References

[1] https://en.wikipedia.org/wiki/Feature_extraction

[2] [http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html](http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html)

[3] Tareen, Shaharyar Ahmed Khan Saleem, Zahra. (2018). A comparative analysis of
SIFT, SURF, KAZE, AKAZE, ORB, and BRISK. 10.1109/ICOMET.2018.8346440.

[4] https://mahotas.readthedocs.io/en/latest/features.html


