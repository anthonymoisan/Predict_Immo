from flask import Flask, jsonify
from flask_restful import reqparse, Api, Resource
from model import Model,TypeRegression
import numpy as np
import pandas as pd
import json
from buildModel import dataCleaning, buildEnsembleX_y, standardisationNumerical
from sklearn.model_selection import train_test_split


app = Flask(__name__)
api = Api(app)

class All_Step_Predict_With_AllModels(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('data', location='form')
        # use parser and find the user's query
        args = parser.parse_args()
        dataset = pd.read_json(json.loads(args['data']))
        print("--> Data Cleaning")
        if 'Column1' in dataset.columns:
            dataset.drop(['Column1'], inplace=True, axis=1)  
        dataCleaning(dataset)
        print("--> Build X and Y")
        X,y = buildEnsembleX_y(dataset)
        print("--> Train and test split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)       
        print("--> Normalisation")
        X_train_scaled,X_test_scaled,scaler = standardisationNumerical(X_train,X_test)
        
        print("--> Random Forest")
        model = Model(TypeRegression.RandomForest)
        model.instanciation()
        print("--> Download model")
        model.load_clf()
        print("--> Predict")
        
        print("Score sur le train : ", model.score(X_train_scaled,y_train))
        print("Score sur le test : ", model.score(X_test_scaled,y_test))
        
        predictionRandomForest = model.predict(scaler.transform(X))
        
        print("--> Régression linéaire")
        model = Model(TypeRegression.RegressionLineaire)
        model.instanciation()
        print("--> Download model")
        model.load_clf()
        print("--> Predict")
        
        print("Score sur le train : ", model.score(X_train_scaled,y_train))
        print("Score sur le test : ", model.score(X_test_scaled,y_test))
        
        predictionRegressionLineaire = model.predict(scaler.transform(X))
       
        print("--> Arbre")
        model = Model(TypeRegression.Arbre)
        model.instanciation()
        print("--> Download model")
        model.load_clf()
        print("--> Predict")
        
        print("Score sur le train : ", model.score(X_train_scaled,y_train))
        print("Score sur le test : ", model.score(X_test_scaled,y_test))
        
        predictionArbre = model.predict(scaler.transform(X))
       
        listTrainOrTest = [None] * (len(y_train)+len(y_test))
    
        for index in y_train.index:
            listTrainOrTest[index] = 1
               
        for index in y_test.index:
            listTrainOrTest[index] = 0


        #dataset.insert(len(dataset.columns),"Train or Test", listTrainOrTest)        
        
        dictResponse = { 
            "id_mutation" : dataset['id_mutation'].tolist(),
            "predictionRandomForest" : predictionRandomForest.tolist(),
            "predictionRegressionLineaire" : predictionRegressionLineaire.tolist(),
            "predictionArbre" : predictionArbre.tolist(),
            "Train or Test": listTrainOrTest,
                     }
        resultDataFrame = pd.DataFrame(dictResponse)
        
        return jsonify(resultDataFrame.to_json())
        

class Predict_RandomForest_ValeurFonciere(Resource):
    
    def post(self):
        
        model = Model(TypeRegression.RandomForest)
        model.instanciation()
        model.load_clf()
        
        parser = reqparse.RequestParser()
        #parser.add_argument(params)
        parser.add_argument('data', location='form')
        
        # use parser and find the user's query
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        
        prediction = model.predict(X)
        return jsonify(prediction.tolist())

class Predict_Arbre_ValeurFonciere(Resource):
    
    def post(self):
        
        model = Model(TypeRegression.Arbre)
        model.instanciation()
        model.load_clf()
        
        parser = reqparse.RequestParser()
        #parser.add_argument(params)
        parser.add_argument('data', location='form')
        
        # use parser and find the user's query
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        
        prediction = model.predict(X)
        return jsonify(prediction.tolist())
    
class Predict_RegressionLineaire_ValeurFonciere(Resource):
    
    def post(self):
        
        model = Model(TypeRegression.RegressionLineaire)
        model.instanciation()
        model.load_clf()
        
        parser = reqparse.RequestParser()
        #parser.add_argument(params)
        parser.add_argument('data', location='form')
        
        # use parser and find the user's query
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        
        prediction = model.predict(X)
        return jsonify(prediction.tolist())

class Cleaning_DataSet(Resource):
    
    def post(self):
        
        parser = reqparse.RequestParser()
        parser.add_argument('data', location='form')
        # use parser and find the user's query
        args = parser.parse_args()
        dataset = pd.read_json(json.loads(args['data']))
        if 'Column1' in dataset.columns:
            dataset.drop(['Column1'], inplace=True, axis=1)
        print("Taille Av", dataset.shape)
        print(dataset.columns)
        dataCleaning(dataset)
        print("Taille Ap", dataset.shape)

        return jsonify(dataset.to_json())

class DefineXY(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('data', location='form')
        # use parser and find the user's query
        
        args = parser.parse_args()
        dataset = pd.read_json(json.loads(args['data']))
        
        X,y = buildEnsembleX_y(dataset)
        
        dictResponse = { 
            "X" : X.to_json(),
            "y": y.to_list()
            }
        
        return jsonify(dictResponse)
        
class TrainSplitTest(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('dataX', location='form')
        parser.add_argument('dataY', location='form')
        # use parser and find the user's query
        
        args = parser.parse_args()
        X = pd.read_json(json.loads(args['dataX']))
        y = np.array(json.loads(args['dataY']))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)       
        
        dictResponse = { 
            "X_train" : X_train.to_json(),
            "X_test": X_test.to_json(),
            "y_train" : y_train.tolist(),
            "y_test" : y_test.tolist()
            }
        
        return jsonify(dictResponse)

class StandardisationNumerical(Resource):
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('dataX_train', location='form')
        parser.add_argument('dataX_test', location='form')
        # use parser and find the user's query
        
        args = parser.parse_args()
        X_train = pd.read_json(json.loads(args['dataX_train']))
        X_test = pd.read_json(json.loads(args['dataX_test']))
        X_train_scaled,X_test_scaled = standardisationNumerical(X_train,X_test)
        
        dictResponse = { 
            "X_train_scaled" : X_train_scaled.tolist(),
            "X_test_scaled": X_test_scaled.tolist()
            }
    
        return jsonify(dictResponse)
    
# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Predict_RandomForest_ValeurFonciere, '/RandomForest')
api.add_resource(Predict_Arbre_ValeurFonciere, '/Arbre')
api.add_resource(Predict_RegressionLineaire_ValeurFonciere, '/RegressionLineaire')
api.add_resource(Cleaning_DataSet, '/CleaningData')
api.add_resource(DefineXY, '/DefineXY')
api.add_resource(TrainSplitTest, '/TrainSplitTest')
api.add_resource(StandardisationNumerical, '/StandardisationNumerical')
api.add_resource(All_Step_Predict_With_AllModels, '/All_Step_Predict_With_AllModels')

if __name__ == '__main__':
    app.run(debug=True)
    

    

