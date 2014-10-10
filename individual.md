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
