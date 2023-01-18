import pandas as pd


def read_csv():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    da = pd.read_csv(r"KDDTrain+Names - 副本.csv")
    return da


if __name__ == "__main__":
    data = read_csv()
    data.info()
