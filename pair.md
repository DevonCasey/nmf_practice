



##LDA and comparison of other techniques
===========================================================

## Use GraphLab

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

## Yelp Revisited

For today's sprint, we will be applying what we have learned yesterday to customer reviews.  This is often a common exploratory application of topic modeling, but can also provide insight into your users.  Yelp has created a novel [wordmap](http://www.yelp.com/wordmap/sf) of which reviews mention certain keywords.  While this application is much simpler than topic modeling (just counts), you could apply topic modeling to the Yelp reviews and plot how much of each topic each neighborhood represents... And we will do exactly that!

1. Using the data in `data/yelp*.json`, apply LDA using gensim to the review text.  What are the 5 most prominent topics?  __NOTE: You may want to try using a subset to start if it is taking long to run__

2. There probably are more than 5 topics for something as varied as reviews. Experiment with a sane number of topics and inspect what is returned.  What is a sensible number of topics?  And what are the most prominent 5 topics of these?

3.  Apply NMF to the same dataset with the same number of topics.  How does NMF compare to LDA?  How good do the topics seem for each?  Are the main topics relatively similar?

4. Time how long each of these takes to run.  Does NMF take longer to run than LDA?  Is the difference in time significant (i.e. long enough for it to be bothersome)?

5. Write your own review and model its topics based on the 'trained' LDA on the corpus of all the Yelp reviews. __HINT: Read the Docs__

6. Repeat the above analysis, but this time we will play with our features.  Instead of using a bag-of-words, use tf-idf weighting.  Also try using n-grams (bi and tri) and inspect the topics.  How do they change with bi/tri-grams?

### Extra

1. Create your own search engine!  Using gensim's ability to find [document similarity](http://radimrehurek.com/gensim/tut3.html), create a restaurant search engine based on the review text.
    * Allow a 'user' to input a string of text: just as an input argument to a function (or a web app if you want to get fancy)
    * return the top 5 restaurants based on the reviews

__Ex: yelp_search("hipster noodles") or yelp_search("late night taco binge")__



##Extra Credit
==========================================================

1. [Word2Vec](http://radimrehurek.com/gensim/models/word2vec.html) is a way of doing topic modeling on individual words where its usage is encoded as a single vector. This can then be used in all sorts of classifiers to do sequential text classification using moving window. 

