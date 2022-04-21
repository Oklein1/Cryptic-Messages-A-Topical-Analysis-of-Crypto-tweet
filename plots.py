import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Uses 3-D KMeans with axes: pos vader score, neg vader score, neu vader score
# Plots and clusters each of tweet by its vader scores, coloring points according to the cluster it was assigned to
def plot_3d_vader_kmeans(df):
    humans = df[~(df['is_bot'] == 1)]
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(humans['vader'].apply(lambda pt: pt[0:3]).tolist())
    humans['kmeans'] = kmeans.labels_

    print("Plotting points...", end='')
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for grp_name, grp_idx in humans.groupby('kmeans').groups.items():

        color = ['blue', 'purple', 'black'][grp_name] # tried to choose colors with no "pos/neg" connotation. grp_name is an index 0, 1, or 2

        pos = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[0]).values
        neg = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[1]).values
        neu = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[2]).values

        ax.scatter(pos, neg, neu, c=color, label=grp_name)

    print("Done. Close plot to continue.")
    plt.show()
    plt.close()


# Same as 3D plot, but with only pos/neg axes
def plot_2d_vader_kmeans(df):
    humans = df[~(df['is_bot'] == 1)]
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(humans['vader'].apply(lambda pt: pt[0:2]).tolist())
    humans['kmeans'] = kmeans.labels_

    import pdb; pdb.set_trace()

    print("Plotting points...", end='')
    fig = plt.figure()
    ax = plt.axes()

    for grp_name, grp_idx in humans.groupby('kmeans').groups.items():

        color = ['blue', 'purple'][grp_name] # tried to choose colors with no "pos/neg" connotation. grp_name is an index 0, 1, or 2

        pos = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[0]).values
        neg = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[1]).values

        ax.scatter(pos, neg, c=color, label=grp_name)

    print("Done. Close plot to continue.")
    plt.show()
    plt.close()


# Plots each tweet on 3 axes: pos vader score, neg vader score, neu vader score
# And colors each tweet according to the class derived from the tweet's compound vader score
def plot_3d_vader_classes(df):
    print("Plotting points...", end='')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for grp_name, grp_idx in df[~(df['is_bot'] == 1)].groupby('class').groups.items():

        color = 'gray'
        label = 'neutral'
        if grp_name == -1: # negative sentiment class
            color = 'red'
            label = 'negative'
        elif grp_name == 1: # positive sentiment class
            color = 'green'
            label = 'positive'

        pos = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[0]).values
        neg = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[1]).values
        neu = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[2]).values

        ax.scatter(pos, neg, neu, c=color, label=label)

    ax.legend()

    print("Done. Close plot to continue.")
    plt.show()
    plt.close()


# Same as 3D plot, but with only pos/neg axes yet all 3 classes colored
def plot_2d_vader_classes(df):
    print("Plotting points...", end='')
    fig = plt.figure()
    ax = plt.axes()
    for grp_name, grp_idx in df[~(df['is_bot'] == 1)].groupby('class').groups.items():

        color = 'gray'
        label = 'Neutral Tweet'
        if grp_name == -1: # negative sentiment class
            color = 'red'
            label = 'Negative Tweet'
        elif grp_name == 1: # positive sentiment class
            color = 'green'
            label = 'Positive Tweet'

        pos = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[0]).values
        neg = df.iloc[grp_idx]['vader'].apply(lambda pt: pt[1]).values

        ax.scatter(pos, neg, c=color, label=label)

    ax.legend()
    plt.xlabel('Negative VADER Score')
    plt.ylabel('Positive VADER Score')

    print("Done. Close plot to continue.")
    plt.savefig('books_read.png')
    plt.show()
    plt.close()