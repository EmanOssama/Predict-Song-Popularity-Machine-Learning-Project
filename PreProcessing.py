from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

def preProcessingByLabelEncoder(data,cols):
    for col in cols:
        labelencoder_x_artist = LabelEncoder();
        data[col] = labelencoder_x_artist.fit_transform(data[col]);
    return  data

def preProcessingByOneHotEncoder(data):
    integer_encoded = np.array(data)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    oneHotEncoding = pd.DataFrame(onehot_encoded)

    return oneHotEncoding
