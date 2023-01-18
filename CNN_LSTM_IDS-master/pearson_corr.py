import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_csv():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    da = pd.read_csv(r"KDDTrain+ calss-5string.csv")
    return da


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal


if __name__ == "__main__":
    data = read_csv()

    sns.heatmap(data.corr(), linewidths=0.1, vmax=1.0, vmin=0.1, square=True, linecolor='white', annot=False, xticklabels=True, yticklabels=True)
    # print(data.corr()['attack_cat'])
    try:
        sys.stdout = Logger('zz.txt')
        print(data.corr())
        # data.corr().to_csv('kdd-heatmap-1v1.csv')
    finally:
        sys.stdout.reset()
    plt.show()
