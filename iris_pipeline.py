from sklearn.datasets import load_iris

def main():
    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Explore dataset
    print("Dataset shape:", df.shape)
    print(df.head())
    print(df.describe())
    print("Class distribution:\n", df['target'].value_counts())
