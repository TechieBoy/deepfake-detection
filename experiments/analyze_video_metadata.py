import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Complete name, Format, Format profile, Codec ID, File size, Duration, Overall bit rate mode, Overall bit rate, Writing application, Video ID, Video Format, Video Format/Info, Video Format profile, Video Format settings, Video Format settings, CABAC, Video Format settings, ReFrames, Video Codec ID, Video Codec ID/Info, Video Duration, Video Bit rate, Video Width, Video Height, Video Display aspect ratio, Video Frame rate mode, Video Frame rate, Video Color space, Video Chroma subsampling, Video Bit depth, Video Scan type, Video Bits/(Pixel*Frame), Video Stream size, Video Writing library, Video Encoding settings, Audio ID, Audio Format, Audio Format/Info, Audio Format profile, Audio Codec ID, Audio Duration, Audio Duration_LastFrame, Audio Bit rate mode, Audio Bit rate, Audio Maximum bit rate, Audio Channel(s), Audio Channel(s)_Original, Audio Channel positions, Audio Sampling rate, Audio Frame rate, Audio Compression mode, Audio Stream size, Audio Default, Audio Alternate group, Modify, Change, Fake

def printStats(df):
    print("\nClass Distribution")
    print(df.groupby( 'Fake' ).size())

    print("\nFake data description")
    print(df[df["Fake"]==1].describe())

    print("\nReal data description")
    print(df[df["Fake"]==0].describe())

    print("\nSkew")
    print(df.skew())

    print("\nPearsons correlation")
    corr = df.corr(method='pearson') # or spearman or kendall
    print(corr)


if __name__ == "__main__":
    df = pd.read_csv("/home/teh_devs/deepfake/raw/video_metadata_analysis.csv")

    print(df.describe())

    printStats(df)