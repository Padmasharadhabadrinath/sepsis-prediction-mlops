import pandas as pd


def load_data(train_path, test_path):

    print("Loading datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    return train_df, test_df
