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
