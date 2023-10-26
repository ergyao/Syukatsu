from enum import IntEnum
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from . import tool

class CoordinatesDataIndex(IntEnum):
    """座標データの並び
    座標データのx座標、y座標、尤度の格納順と1座標あたりのデータ数
    
    X : x座標のインデックス
    
    Y : y座標のインデックス
    
    Lh : 尤度のインデックス
    
    Num : 1座標あたりのデータ数
    """
    X  = 0  #  x座標のインデックス
    Y  = 1  #  y座標のインデックス
    Lh = 2  #  尤度のインデックス
    Num = 3  #  1座標あたりのデータ数

# class CoordinatesPlot:
#     """座標データの確認
    
#     座標データを確認する関数のためのクラス
    
#     """
#     def __init__(
#         self,
#         coordinates: pd.DataFrame,
#         frame_first: int,
#         frame_end: int,
#         save: bool = False,
#         path_save: str = "./",
#         fileName_save: str = "coordinates" 
#     ):
#         self.coordinates = coordinates
        
        

def PlotAllCoordinates(
    coordinates: pd.DataFrame,
    frame_first: int,
    frame_end: int,
    interval:int = 100,
    save: bool = False,
    path_save: str = "./",
    fileName_save: str = "coordinates.gif" 
    
):
    def update(frame):
        ax.cla()  #  軸のリセット？？いる？？
        for part in list(coordinates_parts.keys()):
            #  スケルトンを描画
            ax.plot(coordinates_parts[part].iloc[frame, CoordinatesDataIndex.X.value::CoordinatesDataIndex.Num.value],
                    coordinates_parts[part].iloc[frame, CoordinatesDataIndex.Y.value::CoordinatesDataIndex.Num.value],
                    color=color_parts[part], linewidth=2, alpha=0.2, zorder=1)
            #  特徴点を描画
            ax.scatter(coordinates_parts[part].iloc[frame, CoordinatesDataIndex.X.value::CoordinatesDataIndex.Num.value],
                       coordinates_parts[part].iloc[frame, CoordinatesDataIndex.Y.value::CoordinatesDataIndex.Num.value],
                       color=color_parts[part], s=20, zorder=1)

        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(y_min - 10, y_max + 10)

        ax.set_xlabel("X axis (mm)",fontsize=20)
        ax.set_ylabel("Y axis (mm)",fontsize=20)

    
    coordinates_parts = tool.SeparateIntoParts(coordinates)

    x_min = coordinates.iloc[:, CoordinatesDataIndex.X.value::CoordinatesDataIndex.Num.value].min().min()
    x_max = coordinates.iloc[:, CoordinatesDataIndex.X.value::CoordinatesDataIndex.Num.value].max().max()

    y_min = coordinates.iloc[:, CoordinatesDataIndex.Y.value::CoordinatesDataIndex.Num.value].min().min()
    y_max = coordinates.iloc[:, CoordinatesDataIndex.Y.value::CoordinatesDataIndex.Num.value].max().max()    

    
    color_parts = {"centre": "black",
                   "mantle": "chocolate",
                   "r1"    : "gold",
                   "r2"    : "orange",
                   "r3"    : "red",
                   "r4"    : "magenta",
                   "l1"    : "lime",
                   "l2"    : "seagreen",
                   "l3"    : "turquoise",
                   "l4"    : "blue"}

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    anim = FuncAnimation(fig=fig, func=update, frames=range(frame_first, frame_end), interval=interval)
    
    if save is True:
        store = path_save + fileName_save
        anim.save(store, writer='pillow')
    else:
        plt.show()