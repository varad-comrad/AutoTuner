import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    def __init__(self, data: pd.DataFrame, target_col: str) -> None:
        self.data = data
        self.target_col = target_col

    def describe(self) -> None:
        print(self.data.describe())
        print(self.data.isna().sum())

    def info(self) -> None:
        print(self.data.info())
        print("Shape: ", self.data.shape)
        print("Value Counts of target: ",
              self.data[self.target_col].value_counts())

    def plot(self, kind: str, x: str, y: str) -> None:
        self.data.plot(kind=kind, x=x, y=y)
        plt.show()

    def basic_plots(self, figsize=(10, 5)) -> None:
        for column in self.data.columns:
            if column == self.target_col:
                continue
            if self.data[column].dtype in ['int64', 'float64']:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
                sns.histplot(self.data[column], kde=True, ax=axs[0])
                sns.boxplot(x=self.target_col, y=column,
                            data=self.data, ax=axs[1])
                axs[0].set_title(f'Distribution by {column}')
                axs[0]
                axs[1].set_title(f'{column} vs {self.target_col}')
                fig.show()
            else:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
                sns.countplot(x=column, data=self.data, ax=axs[0])
                sns.countplot(x=column, data=self.data,
                              hue=self.target_col, ax=axs[1])
                axs[0].set_title(f'Distribution by {column}')
                axs[1].set_title(f'{column} vs {self.target_col}')
                fig.show()
