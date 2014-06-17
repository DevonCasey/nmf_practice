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



##LDA and comparison of other techniques
===========================================================

##Gensim

Gensim unfortunately isn't sklearn friendly. Take your corpus again and run it through the mmcorpus. (Below other corpora are listed for posterity). Use this in your model for LDA.
           
           
           '''
           import gensim
           from gensim import corpora, similarities, models


          ##Text Preprocessing is done here using nltk
          
          
          ##Saving of the dictionary and corpus is done here
          ##final_text contains the tokens of all the documents
          
          dictionary = corpora.Dictionary(final_text)
          dictionary.save('questions.dict');
          corpus = [dictionary.doc2bow(text) for text in final_text]
          corpora.MmCorpus.serialize('questions.mm', corpus)
          corpora.SvmLightCorpus.serialize('questions.svmlight', corpus)
          corpora.BleiCorpus.serialize('questions.lda-c', corpus)
          corpora.LowCorpus.serialize('questions.low', corpus)
          
          ##Then the dictionary and corpus can be used to train using LDA
          
          mm = corpora.MmCorpus('questions.mm')
                     '''

[LDA - Latent Dirchlet Association](http://radimrehurek.com/gensim/models/ldamodel.html) is the process of identifying latent topics in your data. LDA allows you to specify the number of topics. The intuition here is that you want to fit an ideal distribution over your sets of words, documents, and topics such that, you identify the likelihood of a word in a given word or topic.

Using LDA, specify a number of topics equal to the new york times articles sections. 

Run through and inspect the clusters. Histogram word counts of the different topics.


Comparisons
============================================================

Now that we have seen LDA, let's run a side by side experiment. Leveraging our LDA results again (as well as the word counts) run through and use:

Remember, kmeans and NMF are from sklearn while LDA is from gensim. Ensure you can pull out comparisons of the data. By that, I mean ensure you can pull out clusters/topics and be able to compare like results.


1. [kmeans](http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf.html#example-applications-topics-extraction-with-nmf-py)
2. LDA (already done)
3. [NMF](http://scikit-learn.org/stable/auto_examples/document_clustering.html#example-document-clustering-py)

Again: histogram word counts and see how each one breaks down per topic. Calculate the number of topics/clusters that are hidden in the word counts and see which method is most effective.






##Extra Credit
==========================================================

1. [Word2Vec](http://radimrehurek.com/gensim/models/word2vec.html) is a way of doing topic modeling on individual words where its usage is encoded as a single vector. This can then be used in all sorts of classifiers to do sequential text classification using moving window. 

2. Extra Credit (Implement NMF using Alternating Least Squares)

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

