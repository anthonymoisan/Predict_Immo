from model import Model,TypeRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Fonction pour lire les donnnées en fonction du fichier
def ReadFile(nomFile, delimiter = '|'):
    # lecture du fichier excel
    df = pd.read_csv(nomFile, delimiter = delimiter, low_memory = False)
    #print("taille du jeu de donnees :", df.shape)
    return df

# Fonction pour extraire les données à partir d'un numéro de département
def ExtractDepartement(df):
    numDep = '75'
    df['code_departement'].astype(str)
    df['Validation'] = (df['code_departement'] == numDep )
    dfDep = df[df['Validation']==True]
    dfDep = dfDep.drop('Validation', axis=1)
    print("Departement : {0}".format(numDep))
    print("Taille du jeu de donnees", dfDep.shape)
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
    #print("Taille suite à agregation :", df.shape)
    
#Methode Remove outliers pour une loi biaisée
def removeOutliers(variable,df):
    #print("avant ", df.shape)
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1
    df.drop(df[(df[variable]<Q1 - 1.5*IQR) | (df[variable]>Q3 + 1.5*IQR)].index, inplace=True)
    #print("après ",df.shape)
    
def readData(file2022,file2021):
    #1) Lecture des donnnées 2022
    df1 = ReadFile(file2022, ',')
    #Lecture des donnnées 2021
    df2 = ReadFile(file2021, ',')
    #Concaténation des deux jeux de données
    df = pd.concat([df1,df2],ignore_index=True)
    #print("taille suite à union :", df.shape)
    #Réduction au département 75
    dataset = ExtractDepartement(df)
    return dataset

def dataCleaning(dataset):
    dataset.drop_duplicates(inplace=True)

    dataset.drop(dataset[(dataset['longitude'].isnull()) | (dataset['latitude'].isnull())].index, inplace=True)
    dataset.drop(dataset[dataset['valeur_fonciere'].isnull() ].index, inplace=True)

    #Suppression des valeurs foncières < 50KE et >3000KE
    dataset.drop(dataset[dataset['valeur_fonciere']<50000 ].index, inplace=True)
    dataset.drop(dataset[dataset['valeur_fonciere']>3000000 ].index, inplace=True)


    #Conversion des objets en string
    dataset['adresse_nom_voie'] = dataset['adresse_nom_voie'].astype("string")
    dataset['adresse_numero'] = dataset['adresse_numero'].astype("string")
    dataset['nom_commune'] = dataset['nom_commune'].astype("string")
    dataset['adresse_complete']=dataset['adresse_numero']+' '+dataset['adresse_nom_voie']+' , '+dataset['nom_commune']  

    dataset.drop(dataset[dataset['type_local']== 'Dépendance' ].index, inplace=True)
    AggregationSimilarData(dataset)

    removeOutliers('valeur_fonciere',dataset)

    dataset["prix m2"]=dataset["valeur_fonciere"]/dataset["surface_reelle_bati"]
    removeOutliers('prix m2',dataset)  
    dataset.drop(["prix m2"],inplace=True,axis=1)
    removeOutliers('lot1_surface_carrez',dataset)

    dataset.drop(dataset[dataset['surface_reelle_bati'].isna()].index, inplace=True, axis=0)

    

def modifyStructure(dataset):
    
    #Suppression des variables qui semblent inutiles
    dataset.drop(['code_departement', 'code_postal', 'adresse_code_voie', 'code_commune', 'id_parcelle','lot1_numero','lot2_numero', 'code_type_local'], inplace=True, axis=1)  
    
    #Suppression des variables qui sont nulles pour 80% des valeurs
    listVariables = dataset.isnull().sum() > (dataset.shape[0]*0.8)
    listResultatsVarDrop = []
    for colname, serie in listVariables.items():
        if(serie == True):
            listResultatsVarDrop.append(colname)
    dataset.drop(listResultatsVarDrop, inplace=True, axis=1)
  
    #Typage et Feature Ingeenering
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
    
    dataset.drop(['nature_mutation'], inplace=True, axis=1)
    dataset.drop(['nombre_pieces_principales'], inplace=True, axis=1)
    dataset.drop(['lot1_surface_carrez'],inplace=True, axis=1)
    dataset["id_mutation"].drop_duplicates(inplace=True)
    dataset.reset_index(inplace=True)
    dataset.drop(['index'], inplace=True, axis = 1)
    
def defineDataSet(file2022,file2021):
    print("Lecture des données")
    dataset = readData(file2022,file2021)
    
    print("Nettoyage des données")
    dataCleaning(dataset)
    
    print("Taille du dataset ", dataset.shape)    
    return dataset

def build_X_y(dataset):
    
    X = dataset.drop(["valeur_fonciere","id_mutation", "date_mutation", "numero_disposition", "adresse_numero", "adresse_nom_voie","adresse_complete"], axis = 1)
    y = dataset["valeur_fonciere"]
    return X,y
    
def dummification(X):
    categorical_features = X.columns[X.dtypes == "category"].tolist()
    df_dummies =  pd.get_dummies(X[categorical_features], drop_first=True)
    X = pd.concat([X.drop(categorical_features, axis=1), df_dummies], axis=1)
    return X
   
def standardisationNumerical(X_train, X_test):
    #centrage des variables numériques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled,X_test_scaled,scaler

def buildEnsembleX_y(dataset):
    modifyStructure(dataset)
    X,y = build_X_y(dataset)
    X = dummification(X)
    return X,y

def build_model(file2022,file2021):
    
    dataset = defineDataSet(file2022,file2021)
    
    X,y = buildEnsembleX_y(dataset)
    
    # #définition des ensembles d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

    X_train_scaled, X_test_scaled = standardisationNumerical(X_train,X_test)
    
    return X,y,X_train_scaled, y_train, X_test_scaled, y_test

def trainAndRun(model, X_train_scaled, y_train, X_test_scaled, y_test):

    model.train(X_train_scaled, y_train)
    print('Model training complete')
    
    print("Score sur le train : ", model.score(X_train_scaled,y_train))
    print("Score sur le test : ", model.score(X_test_scaled,y_test))
    

def downloadModel(model):
    model.pickle_clf()
        
    
if __name__ == "__main__":
        
    
    X,y,X_train_scaled, y_train, X_test_scaled, y_test = build_model("../../input/AvecCoordonneesGeo/full.csv", "../../input/AvecCoordonneesGeo/full2021.csv")
    
    print("\n\nRandom Forest")
    modelRandomForest = Model(TypeRegression.RandomForest)
    modelRandomForest.instanciation()
    trainAndRun(modelRandomForest, X_train_scaled, y_train, X_test_scaled, y_test)
    downloadModel(modelRandomForest)    
        
    print("\n\nArbre de Décision")
    modelArbre = Model(TypeRegression.Arbre)
    modelArbre.instanciation()
    trainAndRun(modelArbre, X_train_scaled, y_train, X_test_scaled, y_test)
    downloadModel(modelArbre)

    print("\n\nRegression Linéaire")
    modelRegression = Model(TypeRegression.RegressionLineaire)
    modelRegression.instanciation()
    trainAndRun(modelRegression, X_train_scaled, y_train, X_test_scaled, y_test)
    downloadModel(modelRegression)