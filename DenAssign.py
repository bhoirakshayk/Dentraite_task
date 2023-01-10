import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder , LabelEncoder


#load json
openjson = open('algoparams_from.txt')
jsondata = openjson.read()
obj = json.loads(jsondata)

features_list = []   #features list

#parse data from json

for k , v in obj["design_state_data"].items():
    if k == "target":
        pred_type = v["prediction_type"]
        target_col = v["target"]

    if k == "feature_handling":
        for i , j in v.items():
            if j["is_selected"] == True :
                features_list.append(j["feature_name"])
    
    if k == "algorithms":
        for m , n in v.items():
            if n["is_selected"] == True:
                algo = m



df = pd.read_csv("iris.csv")
df_selected = df[features_list]  #selecting rows mentioned in json
n_val =dict(df_selected.isnull().sum())  #checking null values

#handling null values
for i,j in n_val.items():
    if j>0:
        from sklearn.impute import SimpleImputer
        if df_selected[i][0] == str:
            si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        else :
            si = SimpleImputer(missing_values=np.nan, strategy='mean')
        si.fit_transform(df_selected[i])

   
x = df_selected.drop(target_col, 1)
y = df_selected[target_col]

#handling object datatype

col = list(x.select_dtypes("object").columns)
oe = OrdinalEncoder()
x[col] = oe.fit_transform(x[col])

if type(y[0]) == str :
    le = LabelEncoder()
    y = le.fit_transform(y)

#problem type
    
if pred_type == "Regression" :
    from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=1)

    def mlmodel(model):
        model.fit(xtrain,ytrain)
        ypred = model.predict(xtest)
        
        train = model.score(xtrain,ytrain)
        test = model.score(xtest,ytest)
        print(f"Training Accuracy : {train}\nTesting Accuracy : {test}\n\n")
        
        print(f"MAE : {mean_absolute_error(ytest,ypred)}")
        print(f"MSE : {mean_squared_error(ytest,ypred)}")
        print(f"Accuracy : {r2_score(ytest,ypred)}")
        return model

    if algo == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        model = mlmodel(RandomForestRegressor())
    
        parameter ={
            "criterion":["squared_error", "absolute_error", "friedman_mse"],
            "max_depth":list(range(1,20)),
            "min_samples_leaf": list(range(1,20)),
            "max_features": ["sqrt", "log2", "None"]
        }

    elif algo == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        model = mlmodel(LinearRegression())
        
        parameter = {
            "fit_intercept": [True, False],
            "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }

    elif algo == "RidgeRegression":
        from sklearn.linear_model import Ridge
        model = mlmodel(Ridge())
        
        parameter = {
            'solver':['svd', 'cholesky', 'lsqr', 'sag'],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
            'normalize':[True, False]
        }
        
    elif algo == "LassoRegression":
        from sklearn.linear_model import Lasso
        model = mlmodel(Lasso())
        
        parameter = {'alpha': np.arange(0.00, 1.0, 0.01)}

    elif algo == "DecisionTreeRegressor":
        from sklearn.tree import DecisionTreeRegressor
        model = mlmodel(DecisionTreeRegressor)
        
        parameter = {
            "criterion":["squared_error", "absolute_error", "friedman_mse"],
            "max_depth":list(range(1,20)),
            "min_samples_leaf": list(range(1,20)),
            "max_features": ["sqrt", "log2", "None"]
        }

    elif algo == "SVM":
        from sklearn.svm import SVR
        model = mlmodel(SVR())
        
        parameter = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf','poly']}
        
    elif algo == "KNN":
        from sklearn.neighbors import KNeighborsregressor
        model = mlmodel(KNeighborsregressor())
        
        parameter = {
            "n_neighbors" : list(range(3,49,3))
        }
        
    elif algo == "extra_random_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = mlmodel(ExtraTreeRegressor())
        
        parameter = {
            "criterion":["squared_error", "absolute_error", "friedman_mse"],
            "max_depth":list(range(1,20)),
            "max_features": ["sqrt", "log2", "None"]
        }    
    
elif pred_type == "Classification" :
    from sklearn.metrics import classification_report , accuracy_score
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=1)

    def mlmodel(model):
        model.fit(xtrain,ytrain)
        ypred = model.predict(xtest)
        
        train = model.score(xtrain,ytrain)
        test = model.score(xtest,ytest)
        print(f"Training Accuracy : {train}\nTesting Accuracy : {test}\n\n")
        
        print(f"Accuracy : {accuracy_score(ytest,ypred)}")
        print(classification_report(ytest,ypred))
        return model

    if algo == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        model = mlmodel(RandomForestClassifier())
        
        parameter = {
            "criterion":["gini", "entropy", "logg_loss"],
            "max_depth":list(range(1,20)),
            "min_samples_leaf": list(range(1,20)),
            "max_features": ["sqrt", "log2", "None"]
        }
        
    elif algo == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = mlmodel(LogisticRegression())
        
        parameter = {
            "C":np.logspace(-3,3,7)
        }

    elif algo == "DecisionTreeClassifier":
        from sklearn.tree import DecisionTreeClassifier
        model = mlmodel(DecisionTreeClassifier())
        
        parameter = {
            "criterion":["gini", "entropy", "logg_loss"],
            "min_samples_leaf": list(range(1,20)),
            "max_features": ["sqrt", "log2", "auto"]
        }
        
    elif algo == "SVM":
        from sklearn.svm import SVC
        model = mlmodel(SVC())
        
        parameter = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf','poly']} 

    elif algo == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = mlmodel(KNeighborsClassifier())
        
        parameter = {
            "n_neighbors" : list(range(3,49,3)),
            "algorithm" : ["auto","ball_tree","kd_tree","brute"]
        }
        
    elif algo == "extra_random_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        model = mlmodel(ExtraTreesClassifier)
        
        parameter = {
            "criterion":["gini", "entropy", "logg_loss"],
            "min_samples_leaf": list(range(1,20)),
            "max_features": ["sqrt", "log2", "None"]
        }

if (obj["design_state_data"]["hyperparameters"]["stratergy"]) == "Grid Search":
    grid = GridSearchCV(model, parameter,verbose=1)
    grid.fit(xtrain, ytrain)
    
    print(grid.best_estimator_)
    mlmodel(grid.best_estimator_)


elif (obj["design_state_data"]["hyperparameters"]["stratergy"]) == "Random Search":
    randm = RandomizedSearchCV(model,parameter)
    randm.fit(xtrain, ytrain)
    
    print(randm.best_estimator_)
    mlmodel(randm.best_estimator_)
