import pandas as pd


def compute_forward_returns(price: pd.DataFrame, periods: int = 1) -> pd.DataFrame():
    """
    只接受股票的调仓日期的价格
    因子日期和股票价格两者的日期需要对应
    横轴为股票代码，纵轴为日期（必须是时间格式）
    :param price: 股票价格数据
    :param periods: 调仓周期
    :return: 收益率数据
    """
    returns = price.pct_change(periods).shift(-periods)
    returns = returns[::periods]
    returns = returns.stack(dropna=True)
    returns.index.set_names(names="stockcode", level=1, inplace=True)
    returns.rename(index=str(periods) + "P", inplace=True)

    return returns


def get_clean_factor(factor: pd.Series, quantiles: int = 10, standard: bool = False) -> pd.DataFrame():
    """
    横轴为因子，纵轴为[日期,股票代码]（日期必须是时间格式）
    warning: 因子值如果有无穷值，会导致标准化会产生未知的精度问题
    :param factor: 因子数据
    :param quantiles: 分层的层数
    :param standard: 是否标准化因子
    :return: 分好组的标准化因子
    """

    def quantile_calc(group, buckets: int):
        return pd.qcut(x=group, q=buckets, labels=False) + 1

    def z_score(group):
        group = (group - group.mean()) / group.std()
        return group

    bucket = factor.groupby("datetime").apply(quantile_calc, buckets=quantiles)
    bucket.rename(index="Group", inplace=True)

    if standard:
        name = factor.name
        factor = factor.groupby("datetime").apply(z_score)
        factor.rename(index=name, inplace=True)

    factor = pd.merge(left=factor, right=bucket, left_index=True, right_index=True, how="left")
    return factor


def merge_returns_factor(forward_return: pd.DataFrame, clean_factor: pd.DataFrame) -> pd.DataFrame():
    """
    将因子组和收益率合并-->数据集
    :param forward_return: compute_forward_returns函数处理后的收益率数据
    :param clean_factor: get_clean_factor函数处理后的因子
    :return: 处理好的因子和收益率数据
    """
    merge_dataset = pd.merge(left=clean_factor, right=forward_return, left_index=True, right_index=True, how="left")
    merge_dataset.dropna(inplace=True)

    return merge_dataset
