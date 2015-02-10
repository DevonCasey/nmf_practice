## Assignment: 

For this assignment, we will apply the NMF algorithm to our corpus of NYT articles to discover latent topics.  The NYT sections are great, but they are somewhat arbitrarily defined.  Let us see what insights we can mine out of our corpus!  



### NMF for the NYT Articles

1. We will be starting with our bag of words matrix.  You may use the [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from scikit-learn (or Tfidf).  You have computed bag of words enough times to know it forwards and backwards.  Use the same 1405 articles we have been using all along now.

2. Use the scikit-learn NMF algorithm to compute the [Non-Negative Matrix factorization](http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf.html) of our documents.  Explore what "topics" are returned. 

3. The output may be hard to understand, but I recommend looking at the top features for each article and also the top words for each feature. Using your vectorizer, extract the feature names into `feature_words` and then the components `H = nmf.components_`, where `nmf` is [sklearn's Non-Negative Matrix Factorization](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) with 15 topics.

#### Interactive Plotting with Plotly.

Make a plotly account following these [instructions] (https://plot.ly/python/getting-started/) and remember your `username` and `api_key`.

Run the following code in your solution with your own username and api_key

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import pandas as pd 
import numpy as np 
import plotly
import plotly.plotly as py
from plotly import graph_objs
py.sign_in('username', 'api_key')
```

1. Make a bar plot of the (top) words for each topic.  The x-axis should represent the word, and the y-axis should represent the value of each word in the topic.  This is similar to looking at the centroids from our kmeans clusters.

    ```python
    trace1 = graph_objs.Bar(
        x=feature_words,
        y=H[0,:],
        name='Finance'
    )
    trace2 = graph_objs.Bar(
        x=feature_words,
        y=H[1,:],
        name='Football'
    )
    data_topics = graph_objs.Data([trace1, trace2])
    layout = graph_objs.Layout(
        title='Word Distributions for Topics of the NYT',
        barmode='group',
        xaxis=graph_objs.XAxis(showticklabels=False, title="Words"),
        yaxis=graph_objs.YAxis(title="Relevance")
    )
    fig = graph_objs.Figure(data=data_topics, layout=layout)
    plot_url = py.plot(fig, filename='nyt_word_distributions', auto_open=False)
    py.iplot(fig, filename='grouped-bar')
    ```

    <div>
        <a href="https://plot.ly/~rickyk9487/2/" target="_blank" title="Word Distributions for Topics of the NYT" style="display: block; text-align: center;"><img src="https://plot.ly/~rickyk9487/2.png" alt="Word Distributions for Topics of the NYT" style="max-width: 100%;"  onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
        <script data-plotly="rickyk9487:2" src="https://plot.ly/embed.js" async></script>
    </div>

1. To really understand the concept of topic space, try choosing a few topics (Politics and Leisure displayed below).  Visualize a small subset of the documents in "topic space" by creating a scatterplot in both two and three dimensions.  Fill your code below so that you use a TfidfVectorizer with arguments `max_df=.8` and `max_features=5000` to do a fit transform on the `content` column of the data. Set `W2` to be the `fit_transform` of this subset of the content.
    
    ```python 
    # 2D code
    data = pd.read_pickle('data/articles.pkl')
    
    ''' 
    Your code here
    '''
    
    traces = []
    for section in data['section_name'].unique():
        trace = dict(type='scatter', 
                     mode='markers', 
                     x=W2[:,0][np.array(data['section_name'] == section)],
                     y=W2[:,1][np.array(data['section_name'] == section)],
                     text = list(heads[data['section_name'] == section]),
                     opacity = 0.8,
                     showlegend = True,
                     name = section,
                     )
    
        traces.append(trace)
    
    x_axis = dict(title='Politics')
    y_axis = dict(title='Leisure')
    layout = dict(title='NYT Projected into 2D Topic Space',
                  xaxis=x_axis,
                  yaxis=y_axis,
                  )
    
    fig = dict(data=traces, layout=layout)
    py.iplot(fig, validate=False)
    ```

    <div>
        <a href="https://plot.ly/~rickyk9487/10/" target="_blank" title="NYT Projected into 2D Topic Space" style="display: block; text-align: center;"><img src="https://plot.ly/~rickyk9487/10.png" alt="NYT Projected into 2D Topic Space" style="max-width: 100%;"  onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
        <script data-plotly="rickyk9487:10" src="https://plot.ly/embed.js" async></script>
    </div>

    ```python
    # 3D code
    def topic_parse(H, n_topics):
    topics_dicts = []

    for i in xrange(n_topics):
        # n_top_words of keys and values
        keys, values = zip(*sorted(zip(feature_words, H[i]), key = lambda x: x[1])[:-n_top_words:-1])
        val_arr = np.array(values)
        norms = val_arr / np.sum(val_arr)
        #normalize = lambda x: int(x / (max(counter.values()) - min(counter.values())) * 90 + 10)
        topics_dicts.append(dict(zip(keys, np.rint(norms* 300))))
    return topics_dicts
    
    small_data = vectorizer.fit_transform(data.content)
    nmf_3 = decomposition.NMF(n_components=3)
    W_3 = nmf_3.fit_transform(small_data)
    H_3 = nmf_3.components_
    
    topics_dicts = topic_parse(H_3, 3)
    
    traces = []
    colors = ["#C659CB",
    "#71C44D",
    "#C25037",
    "#9BA9C1",
    "#454036",
    "#AE4D76",
    "#83C9A3",
    "#D1C046",
    "#6A62AC",
    "#AA8D5C"]
    
    for i, section in enumerate(data['section_name'].unique()):
        indeces = np.array(data['section_name'] == section)
        x = W_3[:,0][indeces] 
        y = W_3[:,1][indeces] 
        z = W_3[:,2][indeces] 
        trace = dict(
            type='scatter3d',
            opacity = 0.8,
            showlegend= True,
            name= section,
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                color= colors[i],
                size=10,
                
            )
        )
        
        traces.append(trace)
      
    layout = dict(
        title='NYT articles projected into 3D Topic Space'   
    )
    
    fig = dict(data=traces, layout=layout)
    
    # labels on axes don't seem to work yet on 3D plots in plotly
    print "X is International Politics"
    print "Y is Sports"
    print "Z is US Government"
    
    from IPython.display import HTML
    
    # legends as well in plotly
    matching = zip(colors, data['section_name'].unique())
    s = "<table style=\"position: relative; bottom: 440px; left: 750px; margin-bottom:-300px\"><td>Color</td><td>Section</td>"
    
    for color, name in matching:
        s += "<tr><td style=\"background: %s\"></td><td>%s</td></tr>" % (color, name)
    
    s += "</table>"
    
    #plot_url = py.plot(fig, filename='nyt_3d_topic', validate=False, auto_open=False)
    HTML(py.iplot(fig, validate=False).data + s)
    ```
    <div>
        <a href="https://plot.ly/~rickyk9487/18/" target="_blank" title="NYT articles projected into 3D Topic Space" style="display: block; text-align: center;"><img src="https://plot.ly/~rickyk9487/18.png" alt="NYT articles projected into 3D Topic Space" style="max-width: 100%;"  onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
        <script data-plotly="rickyk9487:18" src="https://plot.ly/embed.js" async></script>
    </div>


1. Can you add a title to each latent topic representing the words it contains?  Do these make sense given the articles with each topic?

1.  Now that you have hopefully labeled the latent features with what topics they represent, explore a few articles' strongest latent features.  Do these make sense given the article?

1. Compare these results to what your results from kmeans / Hierarchical clustering (Week 5 Day 4).

1. How do the NYT sections compare to the topics from the unsupervised learning?  What are the differences?  And why do you think these exist?

#### Extra:  Word Cloud

Make a word cloud for each latent topic of the words contained in it.  You can use an online service or [Vega](https://github.com/trifacta/vega/blob/master/examples/spec/wordcloud.json) -- an awesome D3 library -- and it's Python library [Vincent](http://vincent.readthedocs.org/en/latest/index.html) (with sweet IPython [bindings](http://vincent.readthedocs.org/en/latest/quickstart.html#ipython-integration)). 

In the terminal, 

  ```
    sudo pip install vincent
    sudo npm install -g d3
  ```

Then in a python environment

  ```python
    
    import vincent
    vincent.core.initialize_notebook()
    for i in xrange(n_topics):
        word_cloud = vincent.Word(topic_dicts[i])
        ''' Your code here'''
        word_cloud.display()
  ```

----------


### Implementing NMF

0. Randomly initialize weights and features matrix
1. Compute difference between __W x H__ and our original matrix using cost function
2. Update __W__ and __H__
3. Repeat steps #1 and #2 until convergence (or max number of iterations)



With the document matrix (our bags of words), we can begin implementing the NMF algorithm.  

1. Create a NMF class to that is initialized with a document matrix (bag of words or tf-idf) __V__.  As arguments (in addition to the document matrix) it should also take parameters __k__ (# of latent topics) and the maximum # of iterations to perform. 
  
  First we need to initialize our weights (__W__) and features (__H__) matrices.  

1. Initialize the weights matrix (W) with (positive) random values to be a __n x k__ matrix, where __n__ is the number of documents and __k__ is the number of latent topics.

2.  Initialize the feature matrix (H) to be __k x m__ where __m__ is the number of words in our vocabulary (i.e. length of bag).  Our original document matrix (__V__) is a __n x m__ matrix.  __NOTICE: shape(V) = shape(W * H)__

3. To make any improvements we need a concept of error. Define a new function to compute the sum of the squared Euclidean distances between each point in our __V_hat__ matrix (__W__ x __H__) and the original document matrix (__V__).

  ![eucl](http://upload.wikimedia.org/math/8/2/0/8206c782235517a0636ff7aa521ed2d7.png)

4. Now that we have initialized our matrices and defined our cost, we can begin iterating. Update your weights and features matrices accordingly.  Update according to the following formulas at each step:

    ![multiplicative_update.png](images/multiplicative_update.png)
    
    This is one of the popular "multiplicative update" rules from a [paper by Lee and Seung](http://hebb.mit.edu/people/seung/papers/nmfconverge.pdf).  
    
    To code this up, use the following.   Note that we update H first, and then feed our H update into the W update.
    Notice also that the operations are elementwise (not our usual linear algebra).  

    ```
                          W.T * R
    H_i+1 = H_i *  --------------------
                        W.T * W * H


                        R * H_i+1.T
    W_i+1 = W_i *  --------------------
                    W * H_i+1 * H_i+1.T
    ```


7. Repeat this update until convergence (i.e. __cost(V, W*H)__ == 0). or until our max # of iterations.

8. Return the computed weights matrix and features matrix.

#### Extra:  Design an API

For extra credit try running your NMF on the book review data from yesterday.  In addition try implementing a user friendly API for your class.  These functions should only return results after you have performed your matrix decomposition.

1. Define a function that displays the top ten words for each of your __k__ topics (and the associated weights).
2. Define a function that displays the headlines/titles of the top 10 documents for each topic.
3. Define a function that takes as input a document and displays the top 3 topics it belongs to.
