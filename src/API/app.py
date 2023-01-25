from flask import Flask, jsonify
from flask_restful import reqparse, Api, Resource
from model import Model,TypeRegression
import numpy as np
import json


app = Flask(__name__)
api = Api(app)

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


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Predict_RandomForest_ValeurFonciere, '/RandomForest')
api.add_resource(Predict_Arbre_ValeurFonciere, '/Arbre')
api.add_resource(Predict_RegressionLineaire_ValeurFonciere, '/RegressionLineaire')


    

if __name__ == '__main__':
    app.run(debug=True)
    

    

