from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import pickle
import enum

class TypeRegression(enum.Enum):
    RandomForest = 1
    Arbre = 2
    RegressionLineaire = 3


class Model(object):

    def __init__(self,TypeRegression):
        self.typeRegression = TypeRegression
        """Mdodel
        Attributes:
            clf: sklearn Random Forest
        """
       
        
    def instanciation(self):
        if(self.typeRegression==TypeRegression.RandomForest):
            nbTree = 500
            depth = 20
            feature = 25
            self.clf = RandomForestRegressor(n_estimators=nbTree, random_state=2, max_depth=depth, max_features=feature)
        elif(self.typeRegression==TypeRegression.Arbre):
            self.clf = DecisionTreeRegressor(max_depth=10)
        elif(self.typeRegression==TypeRegression.RegressionLineaire):
            self.clf = linear_model.LinearRegression()
        

    
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

    def score(self,X,y):
         return self.clf.score(X, y)
    
    def pickle_clf(self):
        """Saves the trained classifier for future use.
        """
        if (self.typeRegression == TypeRegression.RandomForest):
            path = '../Model/RandomForest.pkl'
        elif (self.typeRegression == TypeRegression.Arbre): 
            path = '../Model/Arbre.pkl'
        elif (self.typeRegression == TypeRegression.RegressionLineaire):
            path = '../Model/RegressionLineaire.pkl'
        
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled regression at {}".format(path))
    
    def load_clf(self):
        if (self.typeRegression == TypeRegression.RandomForest):
            path = '../Model/RandomForest.pkl'
        elif (self.typeRegression == TypeRegression.Arbre): 
            path = '../Model/Arbre.pkl'
        elif (self.typeRegression == TypeRegression.RegressionLineaire):
            path = '../Model/RegressionLineaire.pkl'
        
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
            print("Model load at {}".format(path))
