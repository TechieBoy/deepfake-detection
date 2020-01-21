import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn import svm
from sklearn.model_selection import train_test_split

def doPCA(df, n_components=2):
    array = df.values
    x = array[:,:-1]
    y = array[:,-1]
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)
    finalDf = pd.concat([principalDf, pd.DataFrame(y)], axis = 1)
    return finalDf

def scatterPlot(df, sample=100):
    df = df.sample(sample)
    x1 = df.iloc[:, 0]
    x2 = df.iloc[:, 1]
    y = df.iloc[:, 2]
    plt.scatter(x1, x2, c=y)
    return plt

def printStats(df):
    print("\nClass Distribution")
    print(df.groupby( ' Fake' ).size())

    print("\nFake data description")
    print(df[df[" Fake"]==1].describe())

    print("\nReal data description")
    print(df[df[" Fake"]==0].describe())

    print("\nSkew")
    print(df.skew())

    print("\nPearsons correlation")
    corr = df.corr(method='pearson') # or spearman or kendall
    print(corr)

def svmClassifier(df):
    print("Data loaded: ", df.shape)
    array = df.values
    X = array[:,:-1]
    Y = array[:,-1]

    test_size = 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    clf = svm.SVC(kernel='linear', class_weight='balanced')
    print("Training...")
    clf.fit(X_train, Y_train)
    result = clf.score(X_test, Y_test)
    print("Accuracy:", result)
    return clf


if __name__=="__main__":
    names = range(7)
    df_fake = pd.read_csv("~/deepfake/deepfake-detection/experiments/head_pose/feat/fake.txt", sep=" ", names=names).iloc[:, :-1]
    df_real = pd.read_csv("~/deepfake/deepfake-detection/experiments/head_pose/feat/real.txt", sep=" ", names=names).iloc[:, :-1]
    df_fake[" Fake"] = 1
    df_real[" Fake"] = 0
    df = pd.concat([df_fake, df_real], axis=0, ignore_index=True)
    
    # df = doPCA(df)
    svmClassifier(df)

    # printStats(df)

    # scatterPlot(df, 1000)
    # plt.show()