# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:50:07 2023

@author: antho
"""
from buildModel import defineDataSet, build_X_y, standardisationNumerical
from sklearn.model_selection import train_test_split
import json
import requests
import numpy as np

def buildDataSet(filename1,filename2):
    dataset = defineDataSet(filename1,filename2)
    
    X,y = build_X_y(dataset) 
    
    # #définition des ensembles d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)  
    X_train_scaled, X_test_scaled = standardisationNumerical(X_train,X_test)
    return X_test_scaled

def build_1D_Data(X):
    X_Val = X[:][0:1]
    print('Taille de la requete : ',X_Val.shape)
    return X_Val

def build_2D_Data(X):
    print('Taille de la requete : ',X.shape)
    return X

if __name__ == '__main__':
    
    X = buildDataSet("../../input/AvecCoordonneesGeo/full.csv","../../input/AvecCoordonneesGeo/full2021.csv" )
    
    print("\n\n\nRANDOM FOREST")
    print("TEST 1D")
    X_Val = build_1D_Data(X)
    params = {'data': json.dumps(X_Val.tolist())}
    url = 'http://127.0.0.1:5000/RandomForest'
    predict = requests.post(url,params).json()
    print("Prediction : ",np.array(predict))
    
    print("\n")
    print("TEST 2D")
    X_Val2D = build_2D_Data(X)
    params = {'data': json.dumps(X_Val2D.tolist())}
    predict = requests.post(url,params).json()
    print("Prediction : ",np.array(predict))
    
    print("\n\n\nARBRE")
    print("TEST 1D")
    params = {'data': json.dumps(X_Val.tolist())}
    url = 'http://127.0.0.1:5000/Arbre'
    predict = requests.post(url,params).json()
    print("Prediction : ",np.array(predict))
    
    print("\n")
    print("TEST 2D")
    params = {'data': json.dumps(X_Val2D.tolist())}
    predict = requests.post(url,params).json()
    print("Prediction : ",np.array(predict))
    
    print("\n\n\nREGRESSION LINEAIRE")
    print("TEST 1D")
    params = {'data': json.dumps(X_Val.tolist())}
    url = 'http://127.0.0.1:5000/RegressionLineaire'
    predict = requests.post(url,params).json()
    print("Prediction : ",np.array(predict))
    
    print("\n")
    print("TEST 2D")
    params = {'data': json.dumps(X_Val2D.tolist())}
    predict = requests.post(url,params).json()
    print("Prediction : ",np.array(predict))
    