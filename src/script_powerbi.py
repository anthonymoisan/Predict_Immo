import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Fonction pour lire les donnnées en fonction du fichier
def ReadFile(nomFile, delimiter = '|'):
    # lecture du fichier excel
    df = pd.read_csv(nomFile, delimiter = delimiter, low_memory = False)
    print("taille du jeu de donnees :", df.shape)
    return df

# Fonction pour extraire les données à partir d'un numéro de département
def ExtractDepartement(df, numDep):
    df['code_departement'].astype(str)
    df['Validation'] = (df['code_departement'] == numDep )
    dfDep = df[df['Validation']==True]
    dfDep = dfDep.drop('Validation', axis=1)
    print("Departement : {0}".format(numDep))
    print("Taille du jeu de donnees", dfDep.shape)
    print(dfDep.head(5))
    return dfDep  

def AggregationSimilarData(df):
    
    # Construction d'un dictionnaire 
    # où la clé est la chaine de caractère qui permet d'indiquer que deux lignes sont similaires
    # où la valeur est l'index dans le dataframe initial
    dict_similarData = {}
    for index,series in df.iterrows():
        keyRow = str(series['date_mutation'])+'_'+str(series['valeur_fonciere'])+'_'+series['adresse_complete']
        if keyRow in dict_similarData:
            listIndexSimilaire = dict_similarData[keyRow]
            listIndexSimilaire.append(index)
        else:
            listKeyRow = list();
            listKeyRow.append(index)
            dict_similarData[keyRow] = listKeyRow
    
    #Suppression des valeurs dupliquées en prenant comme surface_reelle_bati le cumulé des surfaces
    listIndexASupprimer = []
    for cle,listIndex in dict_similarData.items():
        if(len(listIndex)>1):
            valSurfaceAgregee = df.at[listIndex[0],"surface_reelle_bati"]
            val = 1
            while (val != len(listIndex)):
                listIndexASupprimer.append(listIndex[val])
                valSurfaceAgregee += df.at[listIndex[val],"surface_reelle_bati"]
                val += 1
            df.at[listIndex[0],"surface_reelle_bati"] = valSurfaceAgregee
    #print(listIndexASupprimer)
    df.drop(listIndexASupprimer, inplace = True, axis = 0)
    print("Taille suite à agregation :", df.shape)
    
#Methode Remove outliers pour une loi biaisée
def removeOutliers(variable):
    print("avant ", dfDep.shape)
    Q1 = dfDep[variable].quantile(0.25)
    Q3 = dfDep[variable].quantile(0.75)
    IQR = Q3 - Q1
    dfDep.drop(dfDep[(dfDep[variable]<Q1 - 1.5*IQR) | (dfDep[variable]>Q3 + 1.5*IQR)].index, inplace=True)
    print("après ",dfDep.shape)


# permet de savoir si la ligne fait partie du train ou du test
def retourneEnsembleTrainTest(y_train,y_test):
    #y_train et y_test sont des séries pandas avec index correspondant à la ligne dans y et val le nombre de vélos
    listValue = [None] * (len(y_train)+len(y_test))
    
    for index in y_train.index:
        listValue[index] = 1
       
    for index in y_test.index:
        listValue[index] = 0

    for val in listValue:
        if(not((val==0) or (val==1))):
            print("Error")

    return listValue


def regression_metrics(y, y_pred):
        return pd.DataFrame(
        {
            "max_error": metrics.max_error(y_true=y, y_pred=y_pred),
            "mean_absolute_error": metrics.mean_absolute_error(y_true=y, y_pred=y_pred),
            "mean_squared_error": metrics.mean_squared_error(y_true=y, y_pred=y_pred),
            "r2_score": metrics.r2_score(y_true=y, y_pred=y_pred)
        },
        index=[0])

if __name__ == '__main__':

    #1) Lecture des donnnées 2022
    df1 = ReadFile("../input/AvecCoordonneesGeo/full.csv", ',')
    #Lecture des donnnées 2021
    df2 = ReadFile("../input/AvecCoordonneesGeo/full2021.csv", ',')
    #Concaténation des deux jeux de données
    df = pd.concat([df1,df2])
    print("taille suite à union :", df.shape)
    #Réduction au département 75
    dfDep = ExtractDepartement(df,'75')
    dfDep.to_csv("../input/AvecCoordonneesGeo/75.csv")
    
    #2) Nettoyage des données
    #Suppression des lignes en doublon
    dfDep.drop_duplicates(inplace=True)
    print("taille suite à duplicates :",dfDep.shape)
    
    #Suppression des longitudes et latitudes null
    dfDep.drop(dfDep[(dfDep['longitude'].isnull()) | (dfDep['latitude'].isnull())].index, inplace=True)
    print("taille suite à nettoyage longitude et latitude :",dfDep.shape)
    
    #Suppression des données où la valeur foncière est null
    dfDep.drop(dfDep[dfDep['valeur_fonciere'].isnull() ].index, inplace=True)
    print("taille suite à nettoyage valeur fonciere :",dfDep.shape)
        
    #Suppression des valeurs foncières < 50KE et >3000KE
    dfDep.drop(dfDep[dfDep['valeur_fonciere']<50000 ].index, inplace=True)
    dfDep.drop(dfDep[dfDep['valeur_fonciere']>3000000 ].index, inplace=True)
    print("taille suite à nettoyage valeur fonciere 50KE et 3000KE :",dfDep.shape)
    
    #Suppression des variables qui sont nulles pour 80% des valeurs
    listVariables = dfDep.isnull().sum() > (dfDep.shape[0]*0.8)
    listResultatsVarDrop = []
    for colname, serie in listVariables.iteritems():
        if(serie == True):
                listResultatsVarDrop.append(colname)
    listResultatsVarDrop
    dfDep.drop(listResultatsVarDrop, inplace=True, axis=1)
    print("taille suite à nettoyage lorsque 80% des valeurs sont nulles :",dfDep.shape)

    #Conversion des objets en string
    dfDep['adresse_nom_voie'] = dfDep['adresse_nom_voie'].astype("string")
    dfDep['adresse_numero'] = dfDep['adresse_numero'].astype("string")
    dfDep['nom_commune'] = dfDep['nom_commune'].astype("string")
    dfDep['adresse_complete']=dfDep['adresse_numero']+' '+dfDep['adresse_nom_voie']+' , '+dfDep['nom_commune']    

    #Suppression des données où le type de local est une dépendance
    dfDep.drop(dfDep[dfDep['type_local']== 'Dépendance' ].index, inplace=True)
    print("taille suite à nettoyage dependance :",dfDep.shape)
    
    AggregationSimilarData(dfDep)
    #Suppression des variables qui semblent inutiles
    dfDep.drop(['code_departement', 'code_postal', 'adresse_code_voie', 'code_commune', 'id_parcelle','lot1_numero','lot2_numero', 'code_type_local'], inplace=True, axis=1)
    
    
    #3) Typage et Feature Ingeenering
    dfDep["nature_mutation"] = pd.Categorical(dfDep["nature_mutation"], ordered=False)
    dfDep["type_local"] = pd.Categorical(dfDep["type_local"], ordered=False)
    dfDep["nombre_pieces_principales"] = pd.Categorical(dfDep["nombre_pieces_principales"], ordered=False)
    dfDep["nom_commune"] = pd.Categorical(dfDep["nom_commune"], ordered=False)
    dfDep['date_mutation'] = pd.to_datetime(dfDep['date_mutation'], format='%Y/%m/%d')
    dfDep['id_mutation'] = dfDep['id_mutation'].astype("string")
    dfDep['month']=dfDep["date_mutation"].apply(lambda x: x.month)
    dfDep['day'] = dfDep["date_mutation"].apply(lambda x: x.day)
    dfDep['year'] = dfDep["date_mutation"].apply(lambda x: x.year)
    dfDep["month"] = pd.Categorical(dfDep["month"], ordered=True)
    dfDep["day"] = pd.Categorical(dfDep["day"], ordered=True)
    dfDep["year"]= pd.Categorical(dfDep["year"], ordered=True)
    
    #4) Travail sur les outliers et données significatives/manquantes
    removeOutliers('valeur_fonciere')
    dfDep["prix m2"]=dfDep["valeur_fonciere"]/dfDep["surface_reelle_bati"]
    removeOutliers("prix m2")
    removeOutliers("lot1_surface_carrez")
    dfDep.drop(["prix m2"],inplace=True,axis=1)
    dfDep.drop(['nature_mutation'], inplace=True, axis=1)
    dfDep.drop(['nombre_pieces_principales'], inplace=True, axis=1)
    dfDep.drop(['lot1_surface_carrez'],inplace=True, axis=1)
    dfDep.drop(dfDep[dfDep['surface_reelle_bati'].isna()].index, inplace=True, axis=0)
    dfDep.reset_index(inplace=True)
    
    X = dfDep.drop(["valeur_fonciere","id_mutation", "date_mutation", "numero_disposition", "adresse_numero", "adresse_nom_voie","adresse_complete"], axis = 1)
    y = dfDep["valeur_fonciere"]
    print(f"Shape de X : {X.shape}")
    print(f"Shape de y : {y.shape}")
    categorical_features = X.columns[X.dtypes == "category"].tolist()
    df_dummies =  pd.get_dummies(X[categorical_features], drop_first=True)
    X = pd.concat([X.drop(categorical_features, axis=1), df_dummies], axis=1)    

    
    # #définition des ensembles d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
    print(f"Shape du X_train : {X_train.shape}")
    print(f"Shape du y_train : {y_train.shape}")
    print(f"Shape du X_test : {X_test.shape}")
    print(f"Shape du y_test : {y_test.shape}")

    listTrainOrTest = retourneEnsembleTrainTest(y_train, y_test)

    #centrage des variables numériques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # #Meilleur modèle est RandomForest entre régression linéaire et arbre de décision et on met les paramètres optimisées suite à GridSearch
    nbTree = 500
    depth = 20
    feature = 25
    randomForest = RandomForestRegressor(n_estimators=nbTree, random_state=2, max_depth=depth, max_features=feature)
    randomForest.fit(X_train_scaled, y_train)

    # #insertion à la dernière position du dataframe les prédictions de notre modèle sur la matrice X centrée-réduite
    dfDep.insert(len(dfDep.columns),"Predictions", randomForest.predict(scaler.transform(X)))
    dfDep.insert(len(dfDep.columns), "Train or Test", listTrainOrTest )
    # print(dataset.tail(5))
    y_testPred = randomForest.predict(X_test_scaled)
    y_trainPred = randomForest.predict(X_train_scaled)
    print("Regression metrics pour la forêt aléatoire optimisée for train data")
    print(regression_metrics(y_train, y_trainPred))
    print("Regression metrics pour la forêt aléatoire optimisée for test data")
    print(regression_metrics(y_test, y_testPred))


''' Script Python à insérer dans POWER BI. On enlève la lecture qui est réalisé par Power BI et les métriques et trace
#'dataset' contient les données d'entrée pour ce script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

dataset.drop_duplicates(inplace=True)
dataset.drop(dataset[(dataset['longitude'].isnull()) | (dataset['latitude'].isnull())].index, inplace=True)
dataset.drop(dataset[dataset['valeur_fonciere'].isnull() ].index, inplace=True)
#Suppression des valeurs foncières < 50KE et >3000KE
dataset.drop(dataset[dataset['valeur_fonciere']<50000 ].index, inplace=True)
dataset.drop(dataset[dataset['valeur_fonciere']>3000000 ].index, inplace=True)
#Suppression des variables qui sont nulles pour 80% des valeurs
listVariables = dataset.isnull().sum() > (dataset.shape[0]*0.8)
listResultatsVarDrop = []
for colname, serie in listVariables.iteritems():
    if(serie == True):
        listResultatsVarDrop.append(colname)
dataset.drop(listResultatsVarDrop, inplace=True, axis=1)

#Conversion des objets en string
dataset['adresse_nom_voie'] = dataset['adresse_nom_voie'].astype("string")
dataset['adresse_numero'] = dataset['adresse_numero'].astype("string")
dataset['nom_commune'] = dataset['nom_commune'].astype("string")
dataset['adresse_complete']=dataset['adresse_numero']+' '+dataset['adresse_nom_voie']+' , '+dataset['nom_commune']    

#Suppression des données où le type de local est une dépendance
dataset.drop(dataset[dataset['type_local']== 'Dépendance' ].index, inplace=True)
    
# Construction d'un dictionnaire 
# où la clé est la chaine de caractère qui permet d'indiquer que deux lignes sont similaires
# où la valeur est l'index dans le dataframe initial
dict_similarData = {}
for index,series in dataset.iterrows():
    keyRow = str(series['date_mutation'])+'_'+str(series['valeur_fonciere'])+'_'+series['adresse_complete']
    if keyRow in dict_similarData:
        listIndexSimilaire = dict_similarData[keyRow]
        listIndexSimilaire.append(index)
    else:
        listKeyRow = list();
        listKeyRow.append(index)
        dict_similarData[keyRow] = listKeyRow

#Suppression des valeurs dupliquées en prenant comme surface_reelle_bati le cumulé des surfaces
listIndexASupprimer = []
for cle,listIndex in dict_similarData.items():
    if(len(listIndex)>1):
        valSurfaceAgregee = dataset.at[listIndex[0],"surface_reelle_bati"]
        val = 1
        while (val != len(listIndex)):
            listIndexASupprimer.append(listIndex[val])
            valSurfaceAgregee += dataset.at[listIndex[val],"surface_reelle_bati"]
            val += 1
        dataset.at[listIndex[0],"surface_reelle_bati"] = valSurfaceAgregee
dataset.drop(listIndexASupprimer, inplace = True, axis = 0)

#Suppression des variables qui semblent inutiles
dataset.drop(['code_departement', 'code_postal', 'adresse_code_voie', 'code_commune', 'id_parcelle','lot1_numero','lot2_numero', 'code_type_local'], inplace=True, axis=1)


#3) Typage et Feature Ingeenering
dataset["nature_mutation"] = pd.Categorical(dataset["nature_mutation"], ordered=False)
dataset["type_local"] = pd.Categorical(dataset["type_local"], ordered=False)
dataset["nombre_pieces_principales"] = pd.Categorical(dataset["nombre_pieces_principales"], ordered=False)
dataset["nom_commune"] = pd.Categorical(dataset["nom_commune"], ordered=False)
dataset['date_mutation'] = pd.to_datetime(dataset['date_mutation'], format='%Y/%m/%d')
dataset['id_mutation'] = dataset['id_mutation'].astype("string")
dataset['month']=dataset["date_mutation"].apply(lambda x: x.month)
dataset['day'] = dataset["date_mutation"].apply(lambda x: x.day)
dataset['year'] = dataset["date_mutation"].apply(lambda x: x.year)
dataset["month"] = pd.Categorical(dataset["month"], ordered=True)
dataset["day"] = pd.Categorical(dataset["day"], ordered=True)
dataset["year"]= pd.Categorical(dataset["year"], ordered=True)

Q1 = dataset['valeur_fonciere'].quantile(0.25)
Q3 = dataset['valeur_fonciere'].quantile(0.75)
IQR = Q3 - Q1
dataset.drop(dataset[(dataset['valeur_fonciere']<Q1 - 1.5*IQR) | (dataset['valeur_fonciere']>Q3 + 1.5*IQR)].index, inplace=True)

dataset["prix m2"]=dataset["valeur_fonciere"]/dataset["surface_reelle_bati"]
Q1 = dataset['prix m2'].quantile(0.25)
Q3 = dataset['prix m2'].quantile(0.75)
IQR = Q3 - Q1
dataset.drop(dataset[(dataset['prix m2']<Q1 - 1.5*IQR) | (dataset['prix m2']>Q3 + 1.5*IQR)].index, inplace=True)

Q1 = dataset["lot1_surface_carrez"].quantile(0.25)
Q3 = dataset["lot1_surface_carrez"].quantile(0.75)
IQR = Q3 - Q1
dataset.drop(dataset[(dataset["lot1_surface_carrez"]<Q1 - 1.5*IQR) | (dataset["lot1_surface_carrez"]>Q3 + 1.5*IQR)].index, inplace=True)

dataset.drop(["prix m2"],inplace=True,axis=1)
dataset.drop(['nature_mutation'], inplace=True, axis=1)
dataset.drop(['nombre_pieces_principales'], inplace=True, axis=1)
dataset.drop(['lot1_surface_carrez'],inplace=True, axis=1)
dataset.drop(dataset[dataset['surface_reelle_bati'].isna()].index, inplace=True, axis=0)
dataset.reset_index(inplace=True)
 
X = dataset.drop(["valeur_fonciere","id_mutation", "date_mutation", "numero_disposition", "adresse_numero", "adresse_nom_voie","adresse_complete"], axis = 1)
y = dataset["valeur_fonciere"]


categorical_features = X.columns[X.dtypes == "category"].tolist()
df_dummies =  pd.get_dummies(X[categorical_features], drop_first=True)
X = pd.concat([X.drop(categorical_features, axis=1), df_dummies], axis=1)    

 
# #définition des ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
 
#centrage des variables numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Meilleur modèle est RandomForest entre régression linéaire et arbre de décision et on met les paramètres optimisées suite à GridSearch
nbTree = 500
depth = 20
feature = 25
randomForest = RandomForestRegressor(n_estimators=nbTree, random_state=2, max_depth=depth, max_features=feature)
randomForest.fit(X_train_scaled, y_train)

#insertion à la dernière position du dataframe les prédictions de notre modèle sur la matrice X centrée-réduite
dataset.insert(len(dataset.columns),"Predictions", randomForest.predict(scaler.transform(X)))
    
'''
        