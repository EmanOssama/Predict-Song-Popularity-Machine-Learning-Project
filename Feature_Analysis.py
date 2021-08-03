import matplotlib.pyplot as plt
import seaborn as sns

def getCorrelationRegression(data):
    corr = data.corr()
    # Top 40% Correlation training features with the popularity
    top_feature = corr.index[abs(corr['popularity'] > 0.4)]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

def getCorrelationClassification(data):
    corr = data.corr()
    # Top 40% Correlation training features with the popularity
    top_feature = corr.index[abs(corr['popularity_level'] > 0.4)]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()