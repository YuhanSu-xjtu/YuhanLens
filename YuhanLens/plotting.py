import pandas as pd
from matplotlib import pyplot as plt


def plot_ic(dataset: pd.DataFrame, period: tuple = (3, 1), logger: bool = True):
    """
    计算因子IC序列，作图
    返回作图数据
    :param period: 图表显示持仓时间和调仓周期
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
    plt.show(title="Period={},Hold={}".format(period[0], period[1]))
    ic.plot(kind="density")
    plt.show(title="Period={},Hold={}".format(period[0], period[1]))

    ic_mean = ic.mean()
    ir = ic.mean() / ic.std()
    if logger:
        print("------IC-------")
        print("IC:", ic)
        print("IC_mean", ic_mean)
        print("IR:", ir)
        print("---------------")
    return ic, ic_mean, ir


def plot_quantile(dataset: pd.DataFrame, long_short: bool = False, compare: bool = False, period: tuple = (3, 1),
                  logger: bool = True
                  ):
    """
     因子分层和多空曲线
    :param period: 图表显示持仓时间和调仓周期
    :param compare: 是否只比较最大和最小的净值曲线，当compare=True时
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
        dataset["-1"] = - dataset.iloc[:, 0]

    dataset = dataset + 1
    net = dataset.cumprod()
    end = net.iloc[-1, :]

    net = net.shift(1).fillna(1.0)
    net = net.append(end)
    if compare:
        # 如果compare=True, 多空、中空、bottom空头、bottom、top： 4条净值曲线
        # 否则显示bottom、top：2条净值曲线
        if long_short:
            net = net.iloc[:, [0, net.shape[1] - 4, net.shape[1] - 3, net.shape[1] - 2, net.shape[1] - 1]]
        else:
            net = net.iloc[:, [0, net.shape[1] - 1]]

    net.plot(kind="line", title="Period={},Hold={},Lag={}".format(period[0], period[1], period[2]))
    plt.show()
    if logger:
        print("------净值------")
        print(net)
        print("---------------")

    return net
