import pandas as pd
from matplotlib import pyplot as plt


def plot_ic(dataset: pd.DataFrame, logger: bool = True):
    """
    计算因子IC序列，作图
    返回作图数据
    :param logger: 日志输出
    :param dataset: merge_returns_factor的返回值
    :return: IC序列，IC平均值，IR
    """

    def cal_ic(group, factor: str, returns: str):
        return group[factor].corr(group[returns], method="spearman")

    factor_name = dataset.columns.get_level_values(0)[0]
    return_name = dataset.columns.get_level_values(0)[2]
    ic = dataset.groupby("datetime").apply(cal_ic, factor_name, return_name)
    ic.rename(index="IC", inplace=True)

    ic.plot(kind="line")
    plt.show()
    ic.plot(kind="density")
    plt.show()

    ic_mean = ic.mean()
    ir = ic.mean() / ic.std()
    if logger:
        print("------IC-------")
        print("IC:", ic)
        print("IC_mean", ic_mean)
        print("IR:", ir)
        print("---------------")
    return ic, ic_mean, ir


def plot_quantile(dataset: pd.DataFrame, long_short: bool = False, logger: bool = True):
    """
     因子分层和多空曲线
    :param logger: 日志输出
    :param dataset: merge_returns_factor得到的处理好的因子和收益率数据集
    :param long_short:是否计算多空组合
    :return: 净值曲线
    """
    returns = dataset.columns.get_level_values(level=0)[-1]
    dataset = dataset.groupby(["datetime", "Group"])[returns].mean()
    dataset = dataset.unstack()
    dataset.columns.set_names(names=None, inplace=True)
    if long_short:
        dataset["TopBottom"] = dataset.iloc[:, len(dataset.columns) - 1] - dataset.iloc[:, 0]
        dataset["MidBottom"] = dataset.iloc[:, int(len(dataset.columns) / 2) - 1] - dataset.iloc[:, 0]

    dataset = dataset + 1
    net = dataset.cumprod()
    end = net.iloc[-1, :]

    net = net.shift(1).fillna(1.0)
    net = net.append(end)

    net.plot(kind="line")
    plt.show()
    if logger:
        print("------净值------")
        print(net)
        print("---------------")

    return net
