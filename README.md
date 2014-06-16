## Overview -- Unsupervised learning: Finding important features (NMF)

This sprint will use Non-Negative Matrix factorization (NMF) to discover topics from our NYT corpus.  Similar to kmeans and hierarchical clustering, NMF is a technique to help discover latent properties (features) in our data that a human might not have been able to see otherwise. 

## Goals

* Matrix factorization
* Dimensionality reduction
* Latent properties
* Linear combination of features
* LDA

## Assignment: 

For this assignment, we will apply the NMF algorithm to our corpus of NYT articles to discover latent topics.  The NYT sections are great, but they are somewhat arbitrarily defined.  Let us see what insights we can mine out of our corpus!  Once we have gotten a feel for latent properties of our data (topics of articles in this case), we will revisit our recommender and apply a SVD matrix factorization to it.

### The Latent Factor Model

#### Model Overview

The central dogma in constructing a recommendation system using collaborative filtering is that *similar users will rate similar { products | restaurants | movies } similarly*. In our previous sprint, we explicitly encoded this idea by using a similarity function to identify similar products. In this sprint, we will use a model that allows us to identify both similar users and similar { products | restaurants | movies } as a function of **latent factors**.

We can think of latent factors as properties of movies (e.g., actors and actresses, genre, comedic style, etc.) that users have a positive or negative preference for. We do not observe these factors or the users' preferences directly, but we assume that they affect how users tend to rate movies. Note that if users have similar preferences, then according to the model, they will behave similarly, and likewise, if movies have similar latent factors, they will be rated similarly by similar users. Latent factors thus give us an intuitive way to specify a generative model the obeys the central dogma.

One issue that comes up with latent factor models is determining how many latent factors to include. There may be a number of different unmeasured properties that affect ratings in different ways. We deal with the problem of choosing the number of latent factors to include in the same way we deal with choosing __alpha__ in a Naive Bayes problem (or setting any hyperparameters).

### NYT Articles

1. We will be starting with our bag of words matrix.  You may use the [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from scikit-learn (or Tfidf).  You have computed bag of words enough times to know it forwards and backwards.  Use the same 1405 articles we have been using all along now.

2. Use the scikit-learn NMF algorithm to compute the [Non-negative Matrix factorization](http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf.html) of our documents.  Explore what "topics" are returned. 

3. The output may be hard to understand, but I recommend looking at the top features for each article and also the top words for each feature.

4. Make a word cloud for each latent topic of the words contained in it.  You can use an online service or [Vega](https://github.com/trifacta/vega/blob/master/examples/spec/wordcloud.json) -- an awesome D3 library -- and it's Python library [Vincent](http://vincent.readthedocs.org/en/latest/index.html) (with sweet IPython [bindings](http://vincent.readthedocs.org/en/latest/quickstart.html#ipython-integration)).   __Hint: Look for the `Word` method in Vincent__

5. Can you add a title to each latent topic representing the words it contains?  Do these make sense given the articles with each topic?

6.  Now that you have hopefully labeled the latent features with what topics they represent, explore a few articles strongest latent features?  Do these make sense given the article?

7. Compare these results to what your kmeans and Hierarchical clustering pulled out from the data yesterday.

8. How do the NYT sections compare to the topics from the unsupervised learning?  What are the differences?  And why do you think these exist?

### Movie recommendations (because who doesn't like movies)

![nicolas](http://media.giphy.com/media/10FDsotj2F4Gkg/giphy.gif)

#### Dimensionality Reduction for Data Sparsity -- SVD

A common technique for both imporving the quality of a recommendation as well as handling sparse data is matrix factorization.  We will be using Singular value decomposition ([SVD](http://en.wikipedia.org/wiki/Singular_value_decomposition)) to factor our reviews matrix.  This has the effect of reducing the dimension of our feature space and also uncovers latent features: if we are talking about movie reviews latent features might correspond to movie genres.  A good theoretical overview can be found [here](http://www2.research.att.com/~volinsky/papers/ieeecomputer.pdf).

We will be using the [MovieLens](http://grouplens.org/datasets/movielens/) dataset to try to predict user ratings of movies and recommend good movies to watch (for a given user).  Here is a [notebook](http://nbviewer.ipython.org/github/marcelcaraciolo/big-data-tutorial/blob/master/tutorial/1-Playing-with-Recommender-Systems.ipynb) with examples of how to load data and such.  The data files are on the Time Capsul (`datasets/movielens`) and correspond to 100k, 1M, and 10M reviews.

We will be working with the 100k dateset to start.  Load this into pandas (look to the above notebook if you need help with this).

#### Step Zero (optional) -- Visualize your Matrices!

**References:**

* [Intro to Matplotlib](http://nbviewer.ipython.org/url/raw.github.com/profjsb/python-bootcamp/master/Lectures/05_NumpyMatplotlib/IntroMatplotlib.ipynb)
* [Matplotlib tutorial](matplotlib_tutorial/MatPlotLib_Tutorial.ipynb)
* [Matplotlib Reference](http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/scientific-python-lectures/master/Lecture-4-Matplotlib.ipynb)
* Really cool (and advanced) matrix [vizualization](http://www.cise.ufl.edu/research/sparse/matrices/synopsis/)

**Plot High Dimensional Data (Optional)**

1. Now that you have some experience with basic plots in matplotlib, we can apply this knowledge to our previous recommender exercise. 

2. Reproduce Jonathan's feature vector color map.

![feature_vec](img/feature_matrix.png)

3. To do this, you will need to use two subplots.  Setup a subplot with 1 row and 2 columns.

4. Plot on the left subplot the feature vector for the 1000 product dataset.

5. Add a colorbar.

6. Repeat this process on the right sublot for the 45 product data set.

7. Add the appropriate labels and title.

8. Follow this same procedure for your similarity matrices.

![img/similarity.png](img/similarity.png)

9. Save these figures for posterity.

**(Extra) Viz cred(it)**

Reproduce these plots with [D3.js](http://d3js.org/).

* [example](http://bl.ocks.org/mbostock/388495)
* [Dashing D3](https://www.dashingd3js.com/)
* [D3 Tutorial](http://alignedleft.com/tutorials/d3/)

__Call me over if you would like to start diving into D3__

#### Step One -- Factor your Matrices!

**References:**

* Doing Data Science: p. 204-214
* Machine Learning in Action: p. 280-298 
* Mining of Massive Datasets: p. 277-308
* Programming Collective Intelligence: p. 226-239 (covers NMF rather than SVD)

**SVD**

Let us define:

* __m__: # of users
* __n__: # of items
* __k__: # of latent features (also rank of __M__)

We will be trying to decompose our rating matrix into 3 component matrices:

![http://upload.wikimedia.org/math/7/6/7/7677ab8759ae6bfa0f6e8ba7d2a1d24f.png](http://upload.wikimedia.org/math/7/6/7/7677ab8759ae6bfa0f6e8ba7d2a1d24f.png)

Where **M** is our user-product rating matrix (__m__ x __n__), **U** is our _weights_ matrix (__m__ x __k__), **Sigma** is our singular values matrix (__k__ x __k__), and **V*** is our features matrix (__k__ x __n__).  The larger the singular value (**Sigma** matrix), the more important that latent feature.  Since **U** and **V** are orthoginal to each other, we can rearrange them in order of decreasing singular values.  If we want to reduce the dimension, we simply set a threshold for which columns/rows to cut off (low rank approximation):

![http://www.shermanlab.com/science/CS/IR/DiracFockIR2/DiracFockRiemannAndIR_files/image002.gif](http://www.shermanlab.com/science/CS/IR/DiracFockIR2/DiracFockRiemannAndIR_files/image002.gif)

You can think of the weights matrix as representing how much of each latent feature corresponds to each user, and the features matrix as how much of each item/rating contributes to the latent features.

Here is a [picture](img/lecture.jpg) of one of my previous lectures on this topic.

If you would do not feel strong enough in linear algebra and matrix methods, feel free to use [scikit-learn](http://scikit-learn.org) and it's [implementation](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) of SVD or numpy's [SVD](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html) method.  If you love matrix math (and programming)... feel free to implement a Alternating Least Squares implementation yourself wih numpy: [ALS algorithm](http://www.utd.edu/~herve/Abdi-LeastSquares06-pretty.pdf)

#### Step Two -- Recommend!

Now that we have factored our matrices, we can begin recommending.  In addition to having useful latent features which we can explore, we now have a method to predict a user's rating for an unseen/rated movie.

To perform a recommendation with SVD there are two options:

1. Use the **M-tilda** matrix reconstruction to predict reviews directly.
2. Project our original **M** matrix into a lower dimensional space and run our previous collaborative filter algorithm on this new (lower dimensional) matrix.

**Hint: Use the first method for this assignment**

__M-Tilda prediction matrix__

![http://upload.wikimedia.org/math/4/6/e/46eb929773508dd9f915b7082e93a5e7.png](http://upload.wikimedia.org/math/4/6/e/46eb929773508dd9f915b7082e93a5e7.png)

1. Find out how many singular values you need to keep to capture ~90% of the energy (square of the singular values).  To compute total energy sum the squares of the singular values.

2. Strip off the extra dimensions from your matrices to reduce **U** and **V**.

3. If we reverse the SVD by re-multiplying our low rank approximations (i.e. **U**, **S**, and **V**).  Since we compressed the **U**, **Sigma**, and **V** matrices when we took the low rank approximation, our reconstituted **M** matrix is only approximate: i.e. a prediction.

4. Simply treat this **M-tilda** matrix as our normalized sum matrix.  Sort the review predictions of all the items a given user has not yet recommended.  Take the top n of these.

5.  Split the movie lens dataset into a train and test set.  Perform cross validation on your predicted reviews and see how well you do as you change how many Singular values to keep.

__Lower Dimensional Projection__

We can project our original **M** matrix into a lower dimensional space. To do this multiply the transpose of our original matrix by the **U** and **Sigma** matrices.

**M*** __x__ **U** __x__ **Sigma-tilda** = lower dimensional projection of **M**

To build a collaborative filter, simply use this new reduced **M** matrix as your user-rating matrix.  Using this new matrix, perform the [steps](#feature_matrix) you previous completed in the earlier parts of the exercise to complete a similarity matrix.

#### Scale

Try to perform the above analysis with increasingly larger datasets (MovieLens 1M and MovieLens 10M)


##LDA and comparison of other techniques
===========================================================

[LDA - Latent Dirchlet Association](http://radimrehurek.com/gensim/models/ldamodel.html) is the process of identifying latent topics in your data. LDA allows you to specify the number of topics. The intuition here is that you want to fit an ideal distribution over your sets of words, documents, and topics such that, you identify the likelihood of a word in a given word or topic.

Using LDA, specify a number of topics equal to the new york times articles sections. 

Run through and inspect the clusters. Histogram word counts of the different topics.


##Comparisons
============================================================

Now that we have seen LDA, let's run a side by side experiment. Leveraging our LDA results again (as well as the word counts) run through and use:

1. [kmeans](http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf.html#example-applications-topics-extraction-with-nmf-py)
2. LDA (already done)
3. [NMF](http://scikit-learn.org/stable/auto_examples/document_clustering.html#example-document-clustering-py)

Again: histogram word counts and see how each one breaks down per topic. Calculate the number of topics/clusters that are hidden in the word counts and see which method is most effective.






##Extra Credit
==========================================================

1. [Word2Vec](http://radimrehurek.com/gensim/models/word2vec.html) is a way of doing topic modeling on individual words where its usage is encoded as a single vector. This can then be used in all sorts of classifiers to do sequential text classification using moving window. 

2. ## Extra Credit (Implement NMF using Alternating Least Squares)

### Overview

0. Randomly initialize weights and features matrix
1. Compute difference between __W x H__ and our original matrix using cost function
2. Update __W__ and __H__
3. Repeat steps #1 and #2 until convergence (or max number of iterations)

### Implementation

1. With the document matrix (our bags of words), we can begin implementing the NMF algorithm.  First we need to initialize our weights (__W__) and features (__H__) matrices.  Initialize the weights matrix (W) with random values to be a __n x r__ matrix, where __n__ is the number of documents and __r__ is the number of latent features.  __r__ is a user defined parameter.

3.  Initialize the feature matrix (H) to be __r x m__ where __m__ is the number of words in our vocabulary (i.e. length of bag).  Our original document matrix (__V__) is a __n x m__ matrix.  __NOTICE: shape(V) = shape(W * H)__

4. Now that we have initialized our matrices, we can begin iterating.  For each update step we need to define our cost function.  We will be using the sum of squared Euclidean distances.  Define a new function (__def cost():__) to compute the sum of the squared Euclidean distances between each point in our two matrices.

![eucl](http://upload.wikimedia.org/math/8/2/0/8206c782235517a0636ff7aa521ed2d7.png)

5. Break our of our iteration if we happen to achieve convergence (i.e. __cost(V, W*H)__ == 0).

6. Now we update our weights and features matrices.  Update our feature matrix according to the following formula:

```
  	          transpose(W) * V
Hi+1 = Hi *  --------------------
              transpose(W) * W * H


  	          V * transpose(Hi+1)
Wi+1 = Wi *  --------------------
              W * Hi+1 * transpose(Hi+1)
```

7. Repeat this update until convergence (step #5) or until our max # of iterations.

8. Return the computed weights and features matrix.

