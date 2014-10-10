### Overview

0. Randomly initialize weights and features matrix
1. Compute difference between __W x H__ and our original matrix using cost function
2. Update __W__ and __H__
3. Repeat steps #1 and #2 until convergence (or max number of iterations)

### Implementation

With the document matrix (our bags of words), we can begin implementing the NMF algorithm.  

1. Create a NMF class to that is initialized with a document matrix (bag of words or tf-idf) __R__.  As arguments (in addition to the document matrix) it should also take parameters __k__ (# of latent topics) and the maximum # of iterations to perform. 
  
  First we need to initialize our weights (__W__) and features (__H__) matrices.  

1. Initialize the weights matrix (W) with (positive) random values to be a __n x k__ matrix, where __n__ is the number of documents and __k__ is the number of latent topics.

2.  Initialize the feature matrix (H) to be __k x m__ where __m__ is the number of words in our vocabulary (i.e. length of bag).  Our original document matrix (__R__) is a __n x m__ matrix.  __NOTICE: shape(R) = shape(W * H)__

3. To make any imporvements we need a concept of error. Define a new function to compute the sum of the squared Euclidean distances between each point in our __R_hat__ matrix (__W__ x __H__) and the original document matrix (__R__).

  ![eucl](http://upload.wikimedia.org/math/8/2/0/8206c782235517a0636ff7aa521ed2d7.png)

4. Now that we have initialized our matrices and defined our cost, we can begin iterating. Update your weights and features matrices accordingly.  Update our feature matrix according to the following formula at each step __I__:

```
  	                  W.T * R
H_i+1 = H_i *  --------------------
                    W.T * W * H


  	                R * H_i+1.T
W_i+1 = W_i *  --------------------
                W * H_i+1 * H_i+1.T
```

7. Repeat this update until convergence (i.e. __cost(R, W*H)__ == 0). or until our max # of iterations.

8. Return the computed weights matrix and features matrix.


## Extra Credit

For extra credit try running your NMF on the NYT data from yesterday.  In addition try implementing a user friendly API for your class.  These functions should only return results after you have performed your matrix decomposition.

1. Define a function that displays the top ten words for each of your __k__ topics (and the associated weights).
2. Define a function that displays the headlines/titles of the top 10 documents for each topic.
3. Define a function that takes as input a document and displays the top 3 topics it belongs to.


