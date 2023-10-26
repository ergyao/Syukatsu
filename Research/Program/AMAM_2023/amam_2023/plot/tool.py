import pandas as pd
import numpy as np

def SeparateIntoParts(
    data: pd.DataFrame
):

    """各特徴点の座標データと名前を体の部位ごとに分割

    座標データと名前を部位ごとに分割して辞書型で返す。

    Parameters:
    ----------
    data: pd.DataFrame
        座標データ

    Return:
    ----------
    coordinates_parts: dict[str, str]
        特徴点の名前をまとめた文字配列の辞書
        (key:{centre, mantle, r1, r2, r3, r4, l1, l2, l3, l4})
    name_parts: dict[str, float]
        特徴点の名前をまとめた文字配列の辞書
        (key:{centre, mantle, r1, r2, r3, r4, l1, l2, l3, l4})
    """
    coordinates_parts = {}
    name_parts = {}
    parts = ["centre", "mantle", "r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"]
    for part in parts:
        coordinates_parts[part] = []
        name_parts[part] = []
        for index_column in range(len(data.columns)):
            if part in data.columns[index_column]: #  部位の名前を含む列を辞書に追加
                name_parts[part].append(data.columns[index_column])
        coordinates_parts[part] = data.loc[:, name_parts[part]]
    
    return coordinates_parts


def data_settle(
    features: dict,
    data
):
    '''データの特徴点ごとにnumpyにして保存

    Parameters:
    ----------
    feature:dict
        特徴点の列名をまとめた文字配列の辞書
    data:
        pd.DataFrameのデータ

    Return:
    ----------
    dict
        種類でまとめdataのnumpy配列の辞書
    key
    centre:
        centreの特徴点の列名のnumpy配列
    mantle:
        mantleの特徴点の列名のnumpy配列
    r1, r2, r3, r4, l1, l2, l3, l4
        各腕の特徴点の列名のnumpy配列

    '''
    settle = {}
    for arm in list(["centre", "mantle", "r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"]):
        settle[arm] = data.loc[:, features[arm]].to_numpy().astype(float)

    return settle

def arm_initial(
    num_frame: int
):
    '''各腕に指定したframeの初期値nanの初期配列をまとめた辞書を作成

    Parameters:
    ----------
    num_frame:int
        frame数

    Return:
    ----------
    dict
        初期配列をまとめた辞書
    key
    r1, r2, r3, r4, l1, l2, l3, l4
        各腕の初期配列

    '''
    initial = {}
    for arm in list(["r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"]):
        initial[arm] = np.empty(num_frame)
        initial[arm][:] = np.nan
    return initial


def center_filter(
    array
):
    m = len(array)//2
    m = int(m)
    print(array)
    if array.iloc[m] == 1:
        r = array.mean()
    else:
        r = 0
    print(r)
    return r