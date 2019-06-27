import warnings

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)

class NMF(object):
    def __init__(self, V, latent_topics, max_iter=50, error=0.01):
        """
        Initialize our weights (W) and features (H) matrices
        Initialize the weights matrix (W) with (positive) random values to be a n x k matrix, where n is the number of documents and k is the number of latent topics
        Initialize the feature matrix (H) to be k x m where m is the number of words in our vocabulary (i.e. length of bag).
        """
        self.V = V
        self.k = latent_topics
        self.max_iter = max_iter
        self.W = np.random.choice(1000, size=[self.V.shape[0], self.k])
        self.H = np.random.choice(1000, size=[self.k, self.V.shape[1]])
        self.error = error

    def update_H(self):
            """
            minimizing the sum of squared errors predicting the document matrix
            """
            self.H = lstsq(self.W, self.V)[0]
            self.H[self.H<0] = 0

    def update_W(self):
            """
            Use the same lstsq solver to update W while holding H fixed (T means two)
            """
            W_T = lstsq(self.H.T, self.V.T)[0]
            self.W = W_T.T
            self.W[self.W<0] = 0

    def fit(self):
            """
            Repeat update_H and update_W for max_iter iterations, or until convergence (change in cost(V, W*H) ≈ 0)
            """
            for i in range(self.max_iter):
                if self.cost() > self.error:
                    print("Iteration %d | Current Cost: %f" % (i + 1, self.cost()))
                    self.update_H()
                    self.update_W()
                else:
                    print("Convergence met.")

    def cost(self):
        """
        cost function is mean squared error
        """
        return mean_squared_error(self.V, np.dot(self.W, self.H))

    def key_feat_idx(self):
        keys = np.argsort(self.H)
        return keys

def load_data(file="data/articles.pkl", X_col='content', y_col='section_name'):
    """
    Load, tokenize and vectorize data
    """
    df = pd.read_pickle(file)
    X = df[X_col]
    y = df[y_col]
    vect = CountVectorizer(max_features=5000, stop_words='english')
    tok = vect.fit_transform(X)
    return df, X, y, tok, vect

def main():
    """
    Makes it work
    """
    print()
    print("Loading data and initilizing...")
    df, X, y, tok, vect = load_data()
    feature_name = np.array(vect.get_feature_names())
    section_names = df['section_name'].unique()
    print("Running NMF...")
    print("----------------------------------------------------------------")
    print()
    nmf = NMF(tok.todense(), len(section_names))
    nmf.fit()
    print("----------------------------------------------------------------")
    print("Total cost of model: %f" % (nmf.cost()))
    print()
    top_features = feature_name[np.argsort(nmf.H)]
    top_five_features = top_features[:, :-6:-1]
    print("Top five features (Sorted by Topics from 0-10):")
    print(top_five_features)
    print()

if __name__ == "__main__":
    main()

