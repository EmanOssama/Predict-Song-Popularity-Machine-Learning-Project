from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import time
import pickle

def polynomialRegression(X , Y):
    #split data to train and test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.32,random_state=1);

    #X_validation_train,X_validation_test,Y_validation_train,Y_validation_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1);
    # fit the transformed features to Linear Regression
    poly_features = PolynomialFeatures(degree=2)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X)

    # fit the transformed features to Linear Regression
    #poly_model = linear_model.LinearRegression()
    #poly_model.fit(X_train_poly, Y_validation_train)

    # predicting on training data-set
    #y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    #prediction = poly_model.predict(poly_features.fit_transform(X_validation_test))
    #print('Mean Square Error of Validation Polynomial Regression :', metrics.mean_squared_error(Y_validation_test, prediction))
    #print(f"Accuracy Score of Validation Polynomial Regression : {r2_score(Y_validation_test, prediction) * 100:.1f}%")
    ###################################################################

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, Y_train)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))
    print('Mean Square Error of Test Polynomial Regression :', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of Test Polynomial Regression : {r2_score(Y_test, prediction) * 100:.1f}%")

    # save the model to disk
    filename = 'PolynomialRegression.sav'
    pickle.dump(poly_model, open(filename, 'wb'))
'''
    # load the model from disk
    filename = 'PolynomialRegression.sav'
    # Load from file
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    ypred = pickle_model.predict(X_train_poly)

    start=time.time()
    result = pickle_model.score(X_train_poly, Y)
    print('Mean Square Error of Polynomial Regression :',metrics.mean_squared_error(Y,ypred))
    print('R2Score of Polynomial Regression :', r2_score(Y, ypred))
    print(f"Accuracy Score of Polynomial Regression : {result * 100:.5f}%")
    end=time.time()
    print("Testing Time :", end - start, "Sec")

'''