import pandas as pd


def print_info(df: pd.DataFrame, describe: bool = False) -> None:
    """
    打印dataframe的基本信息
    :param df:希望打印的Dataframe,默认多重索引为[时间,股票代码]
    :param describe: 是否输出描述性信息
    :return: None
    """
    print("-----Dataframe 打印".ljust(30, "-"))
    print(df)
    print("-----Dataframe 信息".ljust(30, "-"))
    print(df.info())
    if describe:
        print("-----Dataframe 描述".ljust(30, "-"))
        print(df.describe())
    print("-----Dataframe 列空值占比".ljust(30, "-"))
    print(df.isna().sum(axis=0) / df.sum(axis=0))
    print("-----Index unique".ljust(30, "-"))
    print(df.index.get_level_values(0).unique())
    print(df.index.get_level_values(0).nunique())
