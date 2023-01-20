from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
from model import Model

app = Flask(__name__)
api = Api(app)

model = Model()

clf_path = 'RandomForest.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)
    print('Model load')
    
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictValeurFonciere(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        #print(user_query)
        # vectorize the user's query and make a prediction
        #prediction = model.predict(uq_vectorized)
        
        
        # create JSON object
        #output = {'prediction': pred_text}

        #return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictValeurFonciere, '/')


if __name__ == '__main__':
    app.run(debug=True)

