import pandas as pd
import matplotlib.pyplot as plt

def scatterPlot(df, sample=100):
    df = df.sample(sample)
    df_real = df[df[" Fake"]==0]
    x1 = df_real.iloc[:, 0]
    x2 = df_real.iloc[:, 1]
    plt.scatter(x1, x2, marker='o')
    
    df_fake = df[df[" Fake"]==1]
    x1 = df_fake.iloc[:, 0]
    x2 = df_fake.iloc[:, 1]
    plt.scatter(x1, x2, marker='x')
    return plt

def oneDPlot(df, sample=100):
    df = df.sample(sample)
    r = df[df[" Fake"]==0].iloc[:, 0]
    plt.plot(r, 'o')
    f = df[df[" Fake"]==1].iloc[:, 0]
    plt.plot(f, 'x')
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
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(corr, vmin=-1, vmax=1)
    # fig.colorbar(cax)
    # plt.show()

def getInsight(df):
    threshold = 35
    above35 = df[df[" Frame_Rate"] > threshold].shape[0]
    real_above35 = df[(df[" Fake"]==0) & (df[" Frame_Rate"] > threshold)].shape[0]
    
    print(f"\nWhen the frame rate is higher than {threshold}:")
    print(f"The video has a {round(real_above35 / above35 * 100, 2)}% chance of being real")
    print(f"Real: {real_above35}, Fake: {above35-real_above35}, Total: {above35}")

    below35 = df[df[" Frame_Rate"] < threshold].shape[0]
    real_below35 = df[(df[" Fake"]==0) & (df[" Frame_Rate"] < threshold)].shape[0]
    
    print(f"\nWhen the frame rate is lower than {threshold}:")
    print(f"The video has a {round(real_below35 / below35 * 100, 2)}% chance of being real")
    print(f"Real: {real_below35}, Fake: {below35-real_below35}, Total: {below35}")

if __name__ == "__main__":
    df = pd.read_csv("~/deepfake/raw/audio_metadata.csv")  
    df.drop(['Folder_Number', ' Audio_Name', ' File_Size', ' Stream_Size'], axis=1, inplace=True)
    
    printStats(df)

    # fig = plt.figure()    
    # plt.subplot(2, 2, 1)
    df.drop([' Sampling_Rate'], axis=1, inplace=True)
    plt = oneDPlot(df, 10000)
    plt.show()
    getInsight(df)
