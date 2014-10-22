##Feature wise learning Rates

Using your gradient descent class. Implement adaptive learning rates for gradient descent. 
This will replace your alpha with a vector of learning rates per feature. 

Here is some pseudo code for it:

(Credit to: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/)

master_stepsize = 1e-2 #for example
fudge_factor = 1e-6 #for numerical stability
historical_grad = 0
w = randn #initialize w
while not converged:
E,grad = computeGrad(w)
historical_grad += g^2
adjusted_grad = grad / (fudge_factor + sqrt(historical_grad))
w = w - master_stepsize*adjusted_grad

Master step size is your original learning rate. 

W is your coefficients.

