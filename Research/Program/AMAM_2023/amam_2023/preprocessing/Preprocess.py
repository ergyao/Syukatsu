from enum import IntEnum
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression


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
    Dimension = 2  #  次元数

def LoadData(
    path: str
):
    '''特徴点座標の生データの読み込み
    
    DeepLabCutで得られた特徴点座標のcsvファイルの読み込み整理

    Parameters:
    ----------
    path:str
        Deeplabcut の特徴点の座標推定データのpath
    
    Returns
    -------
    data: pandas.DataFrame
        特徴点座標の生データ
    '''
    
    print("LoadData Start")
    
    #  データ読み込み
    data = pd.read_csv(path)

    #  ラベルの整理
    n_columns = []
    for bodyparts, coords in zip(data.iloc[0, :], data.iloc[1, :]):
        n_columns.append(f"{bodyparts}_{coords}")
    data.columns = n_columns
    data.index = data.loc[:, "bodyparts_coords"]
    data = data.drop(index=["bodyparts", "coords"])
    data = data.drop("bodyparts_coords", axis=1)
    
    #  float型に変換
    data = data.astype(float)
    data.index = data.index.astype(float)
    
    print("LoadData Done")
    
    print(data)

    return data

def DropFeatuerPoint(
    data: pd.DataFrame,
    Th_Lh: float,
    fs: int,
    Th_data: float
):
    '''特徴点の選別
    
    尤度の低い座標データを欠損値とし、欠損値を補完した後にデータ数の少ない特徴点の座標データを除外

    Parameters:
    ----------
    data: pandas.DataFrame
        特徴点座標の生データ

    Th_Lh: float
        欠損値のしきい値
        
    fs: int
        フレームレート
        
    Th_data: float: float
        データ数のしきい値（"centre"の何割か）
    '''
    
    print("DropFeatuerPoint Start")

    num_featurePt = int(data.shape[1]/CoordinatesDataIndex.Num.value)  #  特徴点の数
    num_frame = data.shape[0]  #  総フレーム数

    #  尤度の低い座標データを欠損値(np.nan)に変換
    for featurePt_index in range(num_featurePt):
        for frame in range(num_frame):
            if data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value] < Th_Lh:  #  しきい値以下の尤度の座標データを欠損値に変更
                data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value]  = np.nan
                data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value]  = np.nan
                data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value] = np.nan
    
    #  時間軸で3次スプライン補完
    data.interpolate(method="spline", order=3, limit_direction="forward", limit=fs, limit_area="inside", inplace=True)

    #  データ不足の特徴点の除外
    num_centreData = data.loc[:, "centre_x"].count()  #  欠損値を除いた中心座標データの数
    drop_featurePt_index = []
    for featurePt_index in range(num_featurePt):  #  各特徴点に対して
        if data.iloc[:, CoordinatesDataIndex.Num.value*featurePt_index].count()/num_centreData < Th_data:  #  中心座標データの数と比べて一定割合以下の特徴点は除外
            drop_featurePt_index.append(featurePt_index)
    for featurePt_index in drop_featurePt_index:
        data.iloc[:, CoordinatesDataIndex.Num.value*featurePt_index + CoordinatesDataIndex.X.value]  = np.nan
        data.iloc[:, CoordinatesDataIndex.Num.value*featurePt_index + CoordinatesDataIndex.Y.value]  = np.nan
        data.iloc[:, CoordinatesDataIndex.Num.value*featurePt_index + CoordinatesDataIndex.Lh.value] = np.nan
    
    print("DropFeatuerPoint Done")
    print(data)

#  特徴点座標のノイズ処理
def ApplyFilter(
    data: pd.DataFrame,
    fs: int,
    fc: int
):
    '''特徴点座標のノイズ処理
    
    5次ローパスフィルターにより高周波のノイズ処理

    Parameters:
    ----------
    data:pandas.DataFrame
        フィルターを適応するデータ
    fs: int
        データのフレームレート
    fc: int
        カットオフ周波数
    '''
    
    print("ApplyFilter Start")

    num_featurePt = int(data.shape[1]/CoordinatesDataIndex.Num.value)  #  特徴点の数
    num_frame = data.shape[0]  #  総フレーム数
    
    _data = data.to_numpy().T  #  座標データをndarrayに変換

    #  各特徴点における欠損値の区間を確認
    switchToNan_frame = {}  #  各特徴点において欠損値でない区間を記録
    for featurePt_index in range(num_featurePt):  #  各特徴点に対して
        switchToNan_frame[featurePt_index] = []
        during_nan = False  #  欠損値の区間である間
        for frame in range(num_frame):
            if ~np.isnan(_data[featurePt_index*CoordinatesDataIndex.Num.value][frame]) and (not during_nan):
                switchToNan_frame[featurePt_index].append(frame)
                during_nan = True
            elif np.isnan(_data[featurePt_index*CoordinatesDataIndex.Num.value][frame]) and during_nan:
                switchToNan_frame[featurePt_index].append(frame)
                during_nan = False

    #  LowPassFilter の適応            
    iir_numerator, iir_denominator = signal.butter(5, fc/(fs/2))  #  LowPassFilter の設計
    for featurePt_index in range(num_featurePt):
        for j in range(len(switchToNan_frame[featurePt_index])):  #  欠損値の区間の数文
            if len(switchToNan_frame[featurePt_index]) == j + 1:  #  切り替わりがない区間
                if len(_data[featurePt_index*CoordinatesDataIndex.Num.value][switchToNan_frame[featurePt_index][j]::]) < 20:  #  データ数が不足する場合はフィルター適応しない
                    break  #  continue????
                
                #  フィルターの適応
                x_data    = signal.filtfilt(iir_numerator, iir_denominator, _data[featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value][switchToNan_frame[featurePt_index][j]::])
                y_data    = signal.filtfilt(iir_numerator, iir_denominator, _data[featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value][switchToNan_frame[featurePt_index][j]::])
                like_data = signal.filtfilt(iir_numerator, iir_denominator, _data[featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value][switchToNan_frame[featurePt_index][j]::])

                data.iloc[switchToNan_frame[featurePt_index][j]::, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value]  = x_data
                data.iloc[switchToNan_frame[featurePt_index][j]::, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value]  = y_data
                data.iloc[switchToNan_frame[featurePt_index][j]::, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value] = like_data

            else:  #  後に切り替わりがある区間
                if len(_data[featurePt_index*CoordinatesDataIndex.Num.value][switchToNan_frame[featurePt_index][j]:switchToNan_frame[featurePt_index][j+1]]) < 20:  #  データ数が不足する場合はフィルター適応しない
                    break  #  continue????
                
                #  フィルターの適応
                x_data    = signal.filtfilt(iir_numerator, iir_denominator, _data[featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value][switchToNan_frame[featurePt_index][j]:switchToNan_frame[featurePt_index][j+1]])
                y_data    = signal.filtfilt(iir_numerator, iir_denominator, _data[featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value][switchToNan_frame[featurePt_index][j]:switchToNan_frame[featurePt_index][j+1]])
                like_data = signal.filtfilt(iir_numerator, iir_denominator, _data[featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value][switchToNan_frame[featurePt_index][j]:switchToNan_frame[featurePt_index][j+1]])

                #  値を座標データへ反映
                data.iloc[switchToNan_frame[featurePt_index][j]:switchToNan_frame[featurePt_index][j+1], featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value]  = x_data
                data.iloc[switchToNan_frame[featurePt_index][j]:switchToNan_frame[featurePt_index][j+1], featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value]  = y_data
                data.iloc[switchToNan_frame[featurePt_index][j]:switchToNan_frame[featurePt_index][j+1], featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value] = like_data
    
    print("ApplyFilter Done")
    print(data)


def RemoveSingularPoint(
    data: pd.DataFrame,
    Th_pixel: float
):
    '''不審な位置座標データを除去
    
    体側の隣接した特徴点から一定間隔離れた特徴点を除去

    Parameters:
    ----------
    data:pandas.DataFrame
        適応するデータ
    
    Th_pixel: float
        一定間隔の値
    '''

    print("RemoveSingularPoint Start")

    num_featurePt = int(data.shape[1]/CoordinatesDataIndex.Num.value)  #  特徴点の数
    num_frame = data.shape[0]  #  総フレーム数

    w = ["centre_x", "r1-01_x", "r2-01_x",
         "r3-01_x",  "r4-01_x", "r5-01_x",
         "r6-01_x",  "r7-01_x", "r8-01_x",
         "l1-01_x",  "l2-01_x", "l3-01_x",
         "l4-01_x",  "l5-01_x", "l6-01_x",
         "l7-01_x",  "l8-01_x", "mantle_x"]

    for featurePt_index in range(num_featurePt):
        p = 0
        for frame in range(num_frame):
            if data.columns.values[featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value] in w:
                break
            if ~np.isnan(data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value]):
                dx = data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value] - data.iloc[frame, (featurePt_index - 1)*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value]
                dy = data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value] - data.iloc[frame, (featurePt_index - 1)*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value]
                if p == 0:
                    p = 1
                dxy = np.sqrt(np.square(dx) + np.square(dy))
                if np.isnan(data.iloc[frame, (featurePt_index-1)*CoordinatesDataIndex.Num.value]):
                    data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value]  = np.nan
                    data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value]  = np.nan
                    data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value] = np.nan
                elif dxy > Th_pixel:
                    data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value]  = np.nan
                    data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value]  = np.nan
                    data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Lh.value] = np.nan
    
    print("RemoveSingularPoint Done")

def ToRealCoordinates(
    data: pd.DataFrame,
    cal_path: str = "cal.pkl",
    num_calibrationRow: int = 3,
    num_calibrationColumn: int = 3,
    HN_CalibrationInterval: int = 200,
    VT_CalibrationInterval: int = 150,
):
    '''実座標に変換
    
    線形変換で特徴点座標を実座標に変換

    Parameters:
    ----------
    cal_path:str
        キャリブレーションのデータのpath
    data_path:str
        preprocessing で処理したデータのpath
    n_row:int
        キャリブレーションの行の数
    n_column:int
        キャリブレーションの列の数
    l_row:int
        キャリブレーションの縦の区切りの長さ（mm）
    l_column:int
        キャリブレーションの横の区切りの長さ（mm）
    store:str
        保存先のパス
    '''
    
    print("ToRealCoordinates Start")

    num_featurePt = int(data.shape[1]/CoordinatesDataIndex.Num.value)  #  特徴点の数
    num_frame = data.shape[0]  #  総フレーム数

    # データの取り込み
    cal = pd.read_pickle(cal_path)
    np_cal  = cal.to_numpy()
    _data = data.to_numpy()
    
    #  キャリブレーションした点の実座標を計算
    realCoordinates_calibrationPoint = np.mgrid[0:num_calibrationColumn*HN_CalibrationInterval:HN_CalibrationInterval, 0:num_calibrationRow*VT_CalibrationInterval:VT_CalibrationInterval].T.reshape(-1, CoordinatesDataIndex.Dimension.value)
    X_realCoordinates_calibrationPoint = realCoordinates_calibrationPoint[:, CoordinatesDataIndex.X.value]
    Y_realCoordinates_calibrationPoint = realCoordinates_calibrationPoint[:, CoordinatesDataIndex.Y.value]

    #  回帰モデルの準備
    model_x = LinearRegression()
    model_x.fit(np_cal, X_realCoordinates_calibrationPoint)

    model_y = LinearRegression()
    model_y.fit(np_cal, Y_realCoordinates_calibrationPoint)

    for frame in range(num_frame):  #  各フレームに対して
        for featurePt_index in range(num_featurePt):  #  各特徴点に対して
            xy_coordinates = np.array([_data[frame][featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value], _data[frame][featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value]]).reshape(1, -1)
            
            #  画像座標のxy座標から実座標のxとy座標を推定
            if ~np.isnan(xy_coordinates).any(axis=1):
                x_realCoordinates = model_x.predict(xy_coordinates)
                y_realCoordinates = model_y.predict(xy_coordinates)
                
                #  値を座標データに反映
                data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value] = x_realCoordinates
                data.iloc[frame, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value] = y_realCoordinates
    
    print("ToRealCoordinates Done")
    print(data)

def SaveCoordinates(
    data: pd.DataFrame,
    path: str = "./",
    file_name: str = "coordinates.pkl"  
):
    """座標データを保存
    指定のpathに座標データを保存

    Parameters
    ----------
    data : pandas.Dataframe
        保存する座標データ
    file_name : str
        ファイル名 (default "coordinates.pkl")
    """
    if path[-1] != "/":
        path = path + "/"
    if path[-4:] != ".pkl":
        file_name = file_name + ".pkl"

    store = path + file_name
    print("save data...")
    data.to_pickle(store)
    print(store+" saved")
    
def Preprocess(
    path_csv: str,
    path_cal: str,
    Th_Lh: float = 0.8,
    fs: int = 30,
    fc: int = 3,
    Th_data: float = 0.3,
    Th_pixel: float = None,
    path_save:str = None,
    save: bool = False
):
    '''特徴点座標の前処理
    
    DeepLabCutで得られた特徴点座標の前処理
    ・データの選別
    ・欠損値の補完
    ・ノイズ除去
    ・不審な座標データの除去（必要であれば）
    ・実座標への変換

    Parameters:
    ----------
    path:str
        Deeplabcut の特徴点の座標推定データのpath
    
    Returns
    -------
    coordinate_data: pandas.DataFrame
        前処理後の特徴点座標
    '''

    coordinats_data = LoadData(path=path_csv)
    if save:
        SaveCoordinates(data=coordinats_data, path=path_save, file_name="raw_coordinates")

    DropFeatuerPoint(data=coordinats_data, Th_Lh=Th_Lh, fs=fs, Th_data=Th_data)
    ApplyFilter(data=coordinats_data, fs=fs, fc=fc)
    # RemoveSingularPoint(data=coordinats_data, Th_pixel=Th_pixel)
    if save:
        SaveCoordinates(data=coordinats_data, path=path_save, file_name="image_coordinates")

    ToRealCoordinates(data=coordinats_data, cal_path=path_cal)
    if save:
        SaveCoordinates(data=coordinats_data, path=path_save, file_name="coordinates")
    return coordinats_data

if __name__ == "__main__":
    Preprocess()