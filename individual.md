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

4. Make a bar plot of the words for each topic.  The x-axis should represent the word, and the y-axis should represent the value of each word in the topic.  This is similar to looking at the centroids from our kmeans clusters.

5. Can you add a title to each latent topic representing the words it contains?  Do these make sense given the articles with each topic?

6.  Now that you have hopefully labeled the latent features with what topics they represent, explore a few articles strongest latent features?  Do these make sense given the article?

7. Compare these results to what your kmeans and Hierarchical clustering pulled out from the data yesterday.

8. How do the NYT sections compare to the topics from the unsupervised learning?  What are the differences?  And why do you think these exist?

#### Extra

Make a word cloud for each latent topic of the words contained in it.  You can use an online service or [Vega](https://github.com/trifacta/vega/blob/master/examples/spec/wordcloud.json) -- an awesome D3 library -- and it's Python library [Vincent](http://vincent.readthedocs.org/en/latest/index.html) (with sweet IPython [bindings](http://vincent.readthedocs.org/en/latest/quickstart.html#ipython-integration)).   __Hint: Look for the `Word` method in Vincent__
