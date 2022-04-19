import os

import pandas as pd


def read_file(path: str) -> pd.Series:
    """
    建议多重索引的单因子文件使用这种读取方式，读取单索引的股票价格数据的时候不需要
    YuhanLens工具包中DataFrame或者Series默认设置：时间(1级索引)、股票代码（2级索引）名称 统一为datetime 和 stockcode
    :param path: 读取单因子文件的路径
    :return: 返回单因子的Series
    """
    factor = pd.read_csv(path, dtype={"stockcode": "str"}, parse_dates=True)
    factor["datetime"] = pd.to_datetime(factor["datetime"])
    factor.set_index(["datetime", "stockcode"], inplace=True)
    factor = factor.iloc[:, 0]
    return factor


def compute_forward_lag_returns(price: pd.DataFrame, periods: int = 1, holding: int = 1,
                                lag: int = 0) -> pd.Series(dtype="float64"):
    """
   只接受股票的调仓日期的价格
   因子日期和股票价格两者的日期需要对应
   横轴为股票代码，纵轴为日期（必须是时间格式）
   :param lag: 收益率滞后。当设置为0时，函数与compute_forward_returns等价
   :param price: 股票价格数据
   :param periods: 调仓周期
   :param holding: 持仓时间
   :return: 收益率数据
    """
    if lag + holding > periods:
        raise ValueError("滞后期+持仓期不能大于调仓期，请重新设置参数")

    returns = price.pct_change(holding).shift(-lag - holding)
    returns = returns[::periods]

    returns = returns.stack(dropna=True)
    returns.index.set_names(names="stockcode", level=1, inplace=True)
    returns.rename(index=str(holding) + "H", inplace=True)

    return returns


def compute_forward_returns(price: pd.DataFrame, periods: int = 1, holding: int = 1) -> pd.Series(dtype="float64"):
    """
    只接受股票的调仓日期的价格
    因子日期和股票价格两者的日期需要对应
    横轴为股票代码，纵轴为日期（必须是时间格式）
    :param price: 股票价格数据
    :param periods: 调仓周期
    :param holding: 持仓时间
    :return: 收益率数据
    """
    if holding > periods:
        raise ValueError("持仓期不能大于调仓期，请重新设置参数")
    returns = price.pct_change(holding).shift(-holding)
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
        """
        因子分组函数
        """
        return pd.qcut(x=group, q=buckets, labels=False) + 1

    def z_score(group):
        """
        因子截面标准化函数
        """
        group = (group - group.mean()) / group.std()
        return group

    if standard:
        name = factor.name
        factor = factor.groupby("datetime").apply(z_score)
        factor.rename(index=name, inplace=True)
        if quantiles is None:
            return factor

    bucket = factor.groupby("datetime").apply(quantile_calc, buckets=quantiles)
    bucket.rename(index="Group", inplace=True)
    factor = pd.merge(left=factor, right=bucket, left_index=True, right_index=True, how="inner")
    return factor


def merge_returns_factor(forward_return: pd.DataFrame, clean_factor: pd.DataFrame) -> pd.DataFrame():
    """
    将因子组和收益率合并-->数据集
    :param forward_return: compute_forward_returns函数处理后的收益率数据
    :param clean_factor: get_clean_factor函数处理后的因子
    :return: 处理好的因子和收益率数据
    """
    merge_dataset = pd.merge(left=clean_factor, right=forward_return, left_index=True, right_index=True, how="inner")
    merge_dataset.dropna(inplace=True)

    return merge_dataset


def get_merge_dataset(price: pd.DataFrame, factor: pd.Series, periods: int = 1, holding: int = 1,
                      lag: int = 0, quantiles: int = 10, standard: bool = False) -> pd.DataFrame():
    """
    通过原始的价格和因子数据直接得出对齐的单因子分析数据集
    :param price: 股票价格数据
    :param factor: 因子数据
    :param periods: 调仓周期
    :param holding: 持仓时间
    :param lag: 收益率滞后。当设置为0时，函数与compute_forward_returns等价
    :param quantiles: 分层的层数
    :param standard: 是否标准化因子
    :return:合并好的单因子分析数据集
    """
    returns = compute_forward_lag_returns(price=price, periods=periods, holding=holding, lag=lag)
    factor = get_clean_factor(factor=factor, quantiles=quantiles, standard=standard)
    merge_dataset = merge_returns_factor(forward_return=returns, clean_factor=factor)
    return merge_dataset


def merge_multi_factors_dataset(factors: list, forward_return: pd.Series) -> pd.DataFrame:
    """
    构造多因子数据集
    :param forward_return: 收益率数据
    :param factors: 需要合并的已标准化的因子Series列表
    :return: 合并好的多因子数据集
    """
    merge_dataset = pd.DataFrame()

    for factor in factors:
        if factor.equals(factors[0]):
            merge_dataset = factor
            continue
        merge_dataset = pd.merge(left=merge_dataset, right=factor, left_index=True, right_index=True, how="left")
    merge_dataset = pd.merge(left=merge_dataset, right=forward_return, left_index=True, right_index=True, how="inner")
    return merge_dataset


def cross_sectional_regression(model, dataset: pd.DataFrame, save_path: str = "") -> bool:
    """
    截面回归，每个截面获得一个模型
    :param save_path: 模型保存路径
    :param model: 回归模型
    :param dataset: 多因子数据集
    :return: 索引为时间，值为该时间的回归模型保存路径
    """

    def regression(group: pd.DataFrame):
        print("横截面回归", model)
        return 0

    def create_dir_not_exist(path):
        if path == "":
            raise ValueError("保存路径不能为空")
        elif not os.path.exists(path):
            os.mkdir(path)

    create_dir_not_exist(path=save_path)
    dataset.groupby("datetime").apply(regression)
    return True


def sliding_window_regression(model, dataset: pd.DataFrame, save_path: str = "", window: int = 3) -> bool:
    """
    滑窗回归，每个滑窗获得一个模型
    :param save_path: 模型保存路径
    :param window: 滑窗长度
    :param model: 回归模型
    :param dataset: 多因子数据集
    :return: 索引为时间，值为该时间的回归模型保存路径
    """

    def regression_save(data: pd.DataFrame):
        print("滑窗回归", model)
        return 0

    def create_dir_not_exist(path):
        if path == "":
            raise ValueError("保存路径不能为空")
        elif not os.path.exists(path):
            os.mkdir(path)

    def sliding_window(data: pd.DataFrame, windows: int = 3) -> list:
        time_index = data.index.get_level_values(level=0).unique()
        rolling_window = [time_index[window_start:window_start + windows] for window_start in
                          range(0, len(time_index) - windows + 1, 1)]
        return rolling_window

    create_dir_not_exist(path=save_path)
    dataset.reset_index(level=1, drop=False, inplace=True)
    sliding = sliding_window(data=dataset, windows=window)

    for i in range(len(sliding)):
        window_dataset = dataset.loc[sliding[i], :]
        regression_save(data=window_dataset)

    return True
