from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time
import pickle

def multiVariableRegression(X , Y):
    #split data to train and test
    #X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.32,random_state=1);

    #X_validation = np.c_[X_train,Y_train]
    #X_validation_train,X_validation_test,Y_validation_train,Y_validation_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1);

    # load the model from disk
    filename = 'MultiVariableRegression.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    ypred=pickle_model.predict(X)

    start=time.time()
    result = pickle_model.score(X, Y)
    print('Mean Square Error of Multivariable Regression :',metrics.mean_squared_error(Y,ypred))
    print('R2Score of Multivariable Regression :', r2_score(Y,ypred))
    print(f"Accuracy Score of Multivariable Regression : {result * 100:.5f}%")
    end=time.time()
    print("Testing Time :", end-start, "Sec")
'''
    cls =linear_model.LinearRegression()
    cls.fit(X_validation_train,Y_validation_train)
    prediction_validation = cls.predict(X_validation_test)
    print('Mean Square Error of Validation Multivariable Regression :', metrics.mean_squared_error(Y_validation_test, prediction_validation))
    print(f"Accuracy Score of Validation Multivariable Regression : {r2_score(Y_validation_test,prediction_validation)*100:.1f}%")

    cls = linear_model.LinearRegression()
    cls.fit(X_train,Y_train)
    prediction= cls.predict(X_test)
    print('Mean Square Error of Test Multivariable Regression :', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of Test Multivariable Regression : {r2_score(Y_test,prediction)*100:.1f}%")

    # save the model to disk
    filename = 'MultiVariableRegression.sav'
    pickle.dump(cls, open(filename, 'wb'))
'''