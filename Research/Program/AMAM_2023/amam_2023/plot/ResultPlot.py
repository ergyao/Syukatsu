import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from . import tool

def PlotGroundedSuckers(
    grounded_suckers: pd.DataFrame,
    arm: str,
    fs: int
):
    """各フレームにおける吸盤の接地情報を可視化
    
    指定した腕の吸盤または"centre"・"mantle"が接地しているか可視化

    Parameters
    ----------
    grounded_suckers : pd.DataFrame
        吸盤の接地情報
    arm : str
        腕（"r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"）の指定もしくは"centre"・"mantle"
    fs : int
        フレームレート
    """
    num_frame = grounded_suckers.shape[0]  #  総フレーム
    grounded_suckers_parts = tool.SeparateIntoParts(grounded_suckers)  #  特徴点のデータを腕ごとにまとめる
    _grounded_suckers = grounded_suckers_parts[arm].replace(0, np.nan)  #  欠損値は0に置換して対象の腕の接地判定を格納
    num_sucker = len(grounded_suckers_parts[arm].columns)  #  吸盤の数
    if arm == "centre" or arm == "mantle":  #  armに"centre"・"mantle"が指定された場合
        y_ticks = [1]
        suckers = [arm]
    else:
        y_ticks = list(range(1, num_sucker+1))  #  y軸の目盛りの位置
        suckers = ["1", "5", "9", "13", "17", "21", "25", "29", "33", "37", "41"]
    
    #  横軸目盛りの設定
    x_ticks = [i for i in range(0, num_frame) if i % (fs/2) == 0]
    time = map(lambda x: x/fs, x_ticks)
    
    #  結果の描画
    fig, ax = plt.subplots(figsize=(10,5))
    for index_sucker in range(num_sucker):  #  各吸盤で（armに"centre","mantle"を指定した場合は各特徴点で）
        ax.axhline(index_sucker + 1, ls = "--", color = "k")  #  補助線の描画
        ax.plot((index_sucker + 1)*_grounded_suckers.iloc[:,index_sucker], lw=5, color = "k")  #  結果の描画
    #  表示範囲・ラベル・目盛り・グリッドの設定
    ax.set_xlim(0, num_frame)
    ax.set_ylabel(f"{arm} Sucker ID",fontsize=20)
    ax.set_xlabel("Time (s)",fontsize=20)
    ax.set_yticks(y_ticks, suckers, fontsize=20)
    ax.set_xticks(x_ticks, time)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='x', color='k', alpha=0.2, lw=2)
    
    plt.show()
    
def PlotArmCurve(
    arm_curve:pd.DataFrame,
    arm: str,
    fs: int
):
    """各フレームにおける腕の屈曲を可視化

    Parameters
    ----------
    arm_curve : pd.DataFrame
        腕の屈曲の評価
    arm : str
        腕（"r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"）の指定
    fs : int
        フレームレート
    """
    num_frame = arm_curve.shape[0]  #  総フレーム
    #  横軸目盛りの設定
    x_ticks = [i for i in range(0, num_frame) if i % (fs/2) == 0]
    time = map(lambda x: x/fs, x_ticks)

    #  結果の描画
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(arm_curve.loc[:,arm])
    #  表示範囲・ラベル・目盛り・グリッドの設定
    ax.set_xlim(0, num_frame)
    ax.set_ylim(0, arm_curve.max().max()+10)
    ax.set_ylabel("MSE ($\mathrm{mm^2}$)",fontsize=20)
    ax.set_xlabel("Time (s)",fontsize=20)
    ax.set_xticks(x_ticks, time)
    ax.grid(axis='x', color='k', alpha=0.2, lw=2)
    
    plt.show()

def PlotDistanceFromCentre(
    distance_fromCentre: pd.DataFrame,
    arm: str,
    index_sucker: int,
    fs: int
):
    """吸盤と中心部の距離を可視化

    Parameters
    ----------
    distance_fromCentre : pd.DataFrame
        吸盤と中心部の距離
    arm : str
        腕（"r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"）の指定
    index_sucker : int
        吸盤のインデックス
    fs : int
        フレームレート
    """
    num_frame = distance_fromCentre.shape[0]  #  総フレーム
    #  横軸目盛りの設定
    x_ticks = [i for i in range(0, num_frame) if i % (fs/2) == 0]
    time = map(lambda x: x/fs, x_ticks)
    
    distance_fromCentre_parts = tool.SeparateIntoParts(distance_fromCentre)
    fig, ax = plt.subplots(figsize=(10,5))
    
    #  結果の描画
    ax.plot(distance_fromCentre_parts[arm].iloc[:, index_sucker])
    #  表示範囲・ラベル・目盛り・グリッドの設定
    ax.set_xlim(0, num_frame)
    ax.set_xticks(x_ticks, time)
    ax.set_xlabel("Time (s)",fontsize=20)
    ax.set_ylabel("Arm Length (mm)",fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='x', color='k', alpha=0.2, lw=2)
    
    plt.show()

def PlotAccelerationPredict(
    acceleration_target,
    acceleration_predict,
    start: int,
    stop: int,
    fs: int,
    num_frame:int
):
    """加速度の予測結果と加速度を表示

    Parameters
    ----------
    acceleration_target : pd.DataFrame
        加速度
    acceleration_predict : pd.DataFrame
        予測した加速度
    start : int
        加速度の予測を開始するフレーム
    stop : int
        加速度の予測を終了するフレーム
    fs : int
        フレームレート
    num_frame : int
        総フレーム数
    """
    fig, ax = plt.subplots()
    ax.plot(acceleration_target)
    nn = np.empty(start)
    nn[:] = np.nan
    _acceleration_predict = np.append(nn, acceleration_predict)
    
    #  結果の描画
    ax.plot(_acceleration_predict)
    
    #  表示範囲・ラベル・目盛り・グリッドの設定
    xticks = [i for i in range(0, num_frame) if i % (fs/2) == 0]
    time = map(lambda x: x/fs, xticks)
    ax.set_xticks(xticks, time)
    ax.set_xlabel("Time (s)",fontsize=20)
    ax.set_ylabel("Acceleration ($\mathrm{mm/s^2}$)",fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlim(start, stop-1)

    plt.show()

def PlotCoefficient(
    grounded_armParts: list[str],
    coefficient: list[float]
):
    """偏回帰係数の表示

    Parameters
    ----------
    grounded_armParts : list[str]
        予測に使用した腕の部位
    coefficient : list[float]
        偏回帰係数
    """
    label_position = range(1, len(grounded_armParts)+ 1)
    fig, ax = plt.subplots()

    ARM = ["R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4"]
    arm = ["r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"]
    print(grounded_armParts)
    for index_armParts, armParts in enumerate(grounded_armParts):
        for index_ARM in range(8):
            if armParts == f"{arm[index_ARM]}_proximal":
                grounded_armParts[index_armParts] = "$\mathrm{%s_{pr}}$"%ARM[index_ARM]
                break
            elif armParts == f"{arm[index_ARM]}_distal":
                grounded_armParts[index_armParts] = "$\mathrm{%s_{dt}}$"%ARM[index_ARM]
                break
    #  結果の描画
    ax.bar(label_position, coefficient, tick_label=grounded_armParts, align="center")
    
    #  表示範囲・ラベル・目盛り・グリッドの設定
    ax.axhline(0, ls = "-", color = "k")
    ax.set_xlabel("Arm ID",fontsize=20)
    ax.set_ylabel("Coefficients",fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18) 

    plt.show()