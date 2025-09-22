from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Explore dataset
    print("Dataset shape:", df.shape)
    print(df.head())
    print(df.describe())
    print("Class distribution:\n", df['target'].value_counts())
