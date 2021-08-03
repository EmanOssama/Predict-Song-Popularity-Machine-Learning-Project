from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from PreProcessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import  numpy as np
import pickle
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import time

def PCA_(dataset):
    cols = ['artists', 'id', 'name', 'release_date','popularity_level']
    dataset = preProcessingByLabelEncoder(dataset, cols)
    dataset1 = dataset.iloc[:, 0:10]
    dataset2 = preProcessingByOneHotEncoder(dataset['key'])
    dataset3 = dataset.iloc[:, 11:19]
    dataset = dataset1.join(dataset2)
    dataset = dataset.join(dataset3)
    dataset = dataset.fillna(0)
    #getCorrelationClassification(spotify_data)
    #dataset.drop(['explicit', 'acousticness', 'artists', 'instrumentalness', 'release_date', 'liveness', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], axis=1, inplace=True)

    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1];
    #split data to train and test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1);

    # load the model from disk

    pcr = make_pipeline(StandardScaler(), PCA(n_components=28), LogisticRegression())
    start=time.time()
    pcr.fit(X_train, Y_train)
    pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline
    prediction = pcr.predict(X_test)
    print('Mean Square Error of PCA :', metrics.mean_squared_error(Y_test, prediction))
    print(f"Accuracy Score of PCA : {pcr.score(X_test, Y_test,sample_weight=None) * 100:.1f}%")
    end=time.time()
    print('Training Time :', end-start, 'Sec')

    # save the model to disk
    filename = 'PCA.sav'
    pickle.dump(pca, open(filename, 'wb'))
    '''    filename = 'PCA.sav'
        # Load from file
        with open(filename, 'rb') as file:
            pickle_model = pickle.load(file)

        start = time.time()
        result = pickle_model.score(X, Y)
        print(f"Accuracy Score of PCA : {result * 100:.5f}%")
        end = time.time()
        print('Testing Time :', end - start, 'Sec')
    '''