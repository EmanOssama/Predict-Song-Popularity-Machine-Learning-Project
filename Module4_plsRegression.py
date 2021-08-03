from sklearn.cross_decomposition import PLSRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time
import pickle

def plsRegression(X , Y):
    #split data to train and test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.32,random_state=1);

    #X_validation = np.c_[X_train,Y_train]
    #X_validation_train,X_validation_test,Y_validation_train,Y_validation_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1)

    pls = PLSRegression(n_components=6)
    #pls.fit(X_validation_train, Y_validation_train)
    #prediction = pls.predict(X_validation_test)
    #print('Mean Square Error of Validation PLS :', metrics.mean_squared_error(Y_validation_test, prediction))
    #print(f"Accuracy Score of Validation PLS Regression : {r2_score(Y_validation_test,prediction)*100:.1f}%")

    pls.fit(X_train, Y_train)
    prediction = pls.predict(X_test)
    print('Mean Square Error of Test PLS :', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of Test PLS Regression : {r2_score(Y_test,prediction)*100:.1f}%")

    # save the model to disk
    filename = 'PLSRegression.sav'
    pickle.dump(pls, open(filename, 'wb'))
'''
    # load the model from disk
    filename = 'PLSRegression.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    ypred = pickle_model.predict(X)

    start=time.time()
    result = pickle_model.score(X, Y)
    print('Mean Square Error of PLS Regression :',metrics.mean_squared_error(Y,ypred))
    print('R2Score of PLS Regression :', r2_score(Y, ypred))
    print(f"Accuracy Score of PLS Regression : {result * 100:.5f}%")
    end=time.time()
    print("Testing Time :", end - start, "Sec")
'''