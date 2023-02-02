# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:50:07 2023

@author: antho
"""
from buildModel import readData, buildEnsembleX_y, standardisationNumerical,dummification,dataCleaning
from sklearn.model_selection import train_test_split
import json
import requests
import numpy as np
import pandas as pd


def buildEnsembleFromDataSet(dataset):
    
    X,y = buildEnsembleX_y(dataset)
    
    # #définition des ensembles d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)  
    
    X_train_scaled, X_test_scaled = standardisationNumerical(X_train,X_test)
    
    return X,y,X_train, X_test, y_train, y_test,X_train_scaled,X_test_scaled

def build_1D_Data(X):
    X_Val = X[:][0:1]
    print('Taille de la requete : ',X_Val.shape)
    return X_Val

def build_2D_Data(X):
    print('Taille de la requete : ',X.shape)
    return X

def AutreTest():
    #Cleaning du dataset
    params = {'data': json.dumps(dataset.to_json()) }
    url = 'http://127.0.0.1:5000/CleaningData'
    stringDataset = requests.post(url,params).json()
    dataset = pd.read_json(stringDataset)
    
    #Define X and y
    params = {'data': json.dumps(dataset.to_json()) }
    url = 'http://127.0.0.1:5000/DefineXY'
    response = requests.post(url,params).json()
    X = pd.read_json(response["X"])
    y = np.array(response["y"])
    
    #Define Train & Test
    params = {'dataX': json.dumps(X.to_json()), 'dataY' : json.dumps(y.tolist())}
    url = 'http://127.0.0.1:5000/TrainSplitTest'
    response = requests.post(url,params).json()
    X_train = pd.read_json(response["X_train"])
    y_train = np.array(response["y_train"])
    X_test = pd.read_json(response["X_test"])
    y_test = np.array(response["y_test"])
    
    #Define standardisation des variables numériques
    params = {'dataX_train': json.dumps(X_train.to_json()), 'dataX_test' : json.dumps(X_test.to_json())}
    url = 'http://127.0.0.1:5000/StandardisationNumerical'
    response = requests.post(url,params).json()
    X_train_scaled = np.array(response["X_train_scaled"])
    X_test_scaled = np.array(response["X_test_scaled"])
    
    print("\n\n\nRANDOM FOREST")
    print("TEST 1D")
    X_Val = build_1D_Data(X_test_scaled)
    url = 'http://127.0.0.1:5000/RandomForest'
    params = {'data': json.dumps(X_Val.tolist())}
    predict = requests.post(url,params).json()
    print("Prediction : ",np.array(predict))
    
    print("\n")
    print("TEST 2D")
    X_Val2D = build_2D_Data(X_test_scaled)
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
    
    
if __name__ == '__main__':
    
    #Lecture du dataset
    dataset = readData("../../input/AvecCoordonneesGeo/full.csv","../../input/AvecCoordonneesGeo/full2021.csv" )
    
    #Réalisation de toutes les étapes sauf lecture du dataset
    params = {'data': json.dumps(dataset.to_json()) }
    url = 'http://127.0.0.1:5000/All_Step_Predict_With_AllModels'
    response = requests.post(url,params).json()
    dataFrameResult = pd.read_json(response)
    print("predictionRandomForest : ",dataFrameResult["predictionRandomForest"])
    print("predictionRegressionLineaire : ",dataFrameResult["predictionRegressionLineaire"])
    print("predictionArbre : ",dataFrameResult["predictionArbre"])
    
    print("id mutation :",dataFrameResult["id_mutation"])
    print("Train or Test :",dataFrameResult['Train or Test'])
    
    