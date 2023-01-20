from sklearn.ensemble import RandomForestRegressor

# from sklearn.ensemble import RandomForestClassifier
import pickle



class Model(object):

    def __init__(self):
        """Mdodel
        Attributes:
            clf: sklearn regression model
        """
        nbTree = 500
        depth = 20
        feature = 25
        self.clf = RandomForestRegressor(n_estimators=nbTree, random_state=2, max_depth=depth, max_features=feature)
        

    
    def train(self, X, y):
        """Trains the regression to associate the label with the sparse matrix
        """
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(X, y)

    
    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    
    def pickle_clf(self, path='RandomForest.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled regression at {}".format(path))
