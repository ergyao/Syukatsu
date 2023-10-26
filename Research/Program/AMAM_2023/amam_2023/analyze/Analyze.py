import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import itertools

import sys
sys.path.append('../')
from preprocessing import Preprocess
from preprocessing.Preprocess import CoordinatesDataIndex
from plot import tool

def CalculateVelocity(
    coordinates: pd.DataFrame,
    fs: int,
    fc: int
):
    """速度の計算
    
    前進差分法によりx軸とy軸の速度を計算
    
    Parameters
    ----------
    coordinates : pd.DataFrame
        座標
    fs : int
        フレームレート
    fc : int
        カットオフ周波数

    Returns
    -------
    df_velocity: pandas.Dataframe
        x軸とy軸の速度
    """

    print("CalculateVelocity Start")

    np_coordinates = coordinates.to_numpy()  #  narray に変換

    #  xy軸の各方向の velocity を計算
    velocity = np.diff(np_coordinates, axis=0) * fs  #  前進差分法

    #  pd.Dataframeに変換
    dataframe_velocity = pd.DataFrame(velocity)
    dataframe_velocity.columns = coordinates.columns

    #  ノイズ除去
    Preprocess.ApplyFilter(dataframe_velocity, fs, fc)  #  ローパスフィルターの適応

    #  尤度を削除
    columns_Lh = range(CoordinatesDataIndex.Lh.value, velocity.shape[1], CoordinatesDataIndex.Num.value)
    dataframe_velocity = dataframe_velocity.drop(columns=dataframe_velocity.columns[columns_Lh])
    
    print("CalculateVelocity Done")
    
    return dataframe_velocity

def CalculateSpeed(
    velocity: pd.DataFrame
):
    
    print("CalculateSpeed Start")
    
    num_featurePt = int(velocity.shape[1]/CoordinatesDataIndex.Dimension.value)  #  特徴点の数
    num_frame = velocity.shape[0]  #  総フレーム数
    
    np_velocity = velocity.to_numpy()  #  narray に変換
    
    speed = np.empty((num_frame, num_featurePt))

    speed_columns = []  #  列名
    for featurePt_index in range(num_featurePt):  #  各特徴点で実行
        speed[:, featurePt_index] = np.sqrt(np.square(np_velocity[:, featurePt_index*CoordinatesDataIndex.Dimension.value + CoordinatesDataIndex.X.value]) + np.square(np_velocity[:, featurePt_index*CoordinatesDataIndex.Dimension.value + CoordinatesDataIndex.Y.value]))
        column = velocity.columns.values[featurePt_index*2]
        column = column[:-2]
        speed_columns.append(f"{column}")

    dataframe_speed = pd.DataFrame(speed)
    dataframe_speed.columns = speed_columns

    print("CalculateSpeed Done")

    return dataframe_speed

def CalculateAcceleration(
    velocity: pd.DataFrame,
    speed: pd.DataFrame,
    fs: int
):
    
    print("CalculateAcceleration Start")
    
    np_speed = speed.to_numpy()
    #  加速度を計算
    acceleration = np.diff(np_speed, axis=0) * fs  #  前進差分法
    dataframe_acceleration = pd.DataFrame(acceleration)
    dataframe_acceleration.columns = speed.columns
    
    np_velocity = velocity.to_numpy()
    #  xy軸の加速度を計算
    acceleration_xy = np.diff(np_velocity, axis=0) * fs  #  前進差分法
    dataframe_acceleration_xy = pd.DataFrame(acceleration_xy)
    dataframe_acceleration_xy.columns = velocity.columns
    
    print("CalculateAcceleration Done")
    
    return dataframe_acceleration, dataframe_acceleration_xy

def EvaluateArmCurve(
    coordinates: pd.DataFrame
):
    """腕の屈曲を評価
    
    各腕の特徴点の単回帰分析における平方二乗誤差を計算し腕の屈曲を評価。
    値が大きいほど屈曲していることを示す

    Parameters
    ----------
    coordinates : pd.DataFrame
        特徴点の座標

    Returns
    -------
    dataframe_armcurve: pandas.DataFrame
        腕の屈曲の評価
    """
    
    print("EvaluateArmCurve Start")

    num_frame = coordinates.shape[0]  #  総フレーム数    
    arms = ["r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"]
    arm_curve = {}
    
    coordinates_parts = tool.SeparateIntoParts(coordinates)  #  特徴点のデータを腕ごとにまとめる
    
    for arm in arms:  #  各腕で
        arm_curve[arm] = np.empty(num_frame)
        for frame in range(num_frame):  #  各フレームで
            
            #  腕の特徴点のx軸とy軸の値をそれぞれ格納
            coordinate_x = coordinates_parts[arm].iloc[frame, CoordinatesDataIndex.X.value::CoordinatesDataIndex.Num.value].to_numpy()
            coordinate_x = coordinate_x[~np.isnan(coordinate_x)].reshape(-1, 1)
            coordinate_y = coordinates_parts[arm].iloc[frame, CoordinatesDataIndex.Y.value::CoordinatesDataIndex.Num.value].to_numpy()
            coordinate_y = coordinate_y[~np.isnan(coordinate_y)].reshape(-1, 1)
            
            #  単回帰分析
            liner_model = LinearRegression()
            liner_model.fit(coordinate_x, coordinate_y)
            y_prediction = liner_model.predict(coordinate_x)
            mse = mean_squared_error(coordinate_y, y_prediction)  #  平方二乗誤差の計算
            arm_curve[arm][frame] = mse
    
    #  データをpandas.DataFrameの形に変換
    dataframe_armcurve = pd.DataFrame(arm_curve)
    dataframe_armcurve.columns = arms
    
    print("EvaluateArmCurve Done")
    
    return dataframe_armcurve

def CaluculateDistanceFromCentre(
    coordinates: pd.DataFrame
):
    """各特徴点から中心部の距離を計算（腕の伸縮の評価）
    各特徴点から中心部（"centre"）のユークリッド距離を計算

    Parameters
    ----------
    coordinates : pd.DataFrame
        座標

    Returns
    -------
    dataframe_distance_fromCentre : pd.DataFrame
        各特徴点から中心部の距離
    """
    
    print("CaluculateDistanceFromCentre Start")
    
    num_featurePt = int(coordinates.shape[1]/CoordinatesDataIndex.Num.value)  #  特徴点の数
    num_frame = coordinates.shape[0]  #  総フレーム数
    
    #  各特徴点から中心部の距離を計算
    distance_fromCentre = np.empty((num_frame, num_featurePt))
    distance_fromCentre_columns = []
    for featurePt_index in range(num_featurePt):
        distance_x = coordinates.iloc[:, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.X.value] - coordinates.loc[:, "centre_x"]  #  特徴点と中心部のx軸方向のずれ
        distance_y = coordinates.iloc[:, featurePt_index*CoordinatesDataIndex.Num.value + CoordinatesDataIndex.Y.value] - coordinates.loc[:, "centre_y"]  #  特徴点と中心部のy軸方向のずれ
        distance_fromCentre[:, featurePt_index] = np.sqrt(np.square(distance_x) + np.square(distance_y))  #  ユークリッド距離の計算
        #  列名を作成
        column = coordinates.columns.values[featurePt_index*CoordinatesDataIndex.Num.value]
        column = column[:-2]
        distance_fromCentre_columns.append(f"{column}")
    
    #  データをpandas.DataFrameの形に変換
    dataframe_distance_fromCentre = pd.DataFrame(distance_fromCentre)
    dataframe_distance_fromCentre.columns = distance_fromCentre_columns
    
    print("CaluculateDistanceFromCentre Done")
    
    return dataframe_distance_fromCentre

def DecideGroundedSuckers(
    speed: pd.DataFrame,
    speed_Lh: float
):
    """接地している吸盤を判定
    
    一定の速度以下の吸盤を接地していると判定
    （接地:1, 非接地:0）

    Parameters
    ----------
    speed : pd.DataFrame
        速さ
    speed_Lh : float
        速さのしきい値

    Returns
    -------
    _type_
        吸盤の接地判定（接地:1, 非接地:0）
    """
    print("DecideGroundedSuckers Start")
    
    num_featurePt = int(speed.shape[1])  #  特徴点の数
    num_frame = speed.shape[0]  #  総フレーム数
    np_speed = speed.to_numpy()  #  narrayに変換
    grounded_suckers = np.zeros_like(np_speed)  #  解析結果の保存先を用意
    
    #  接地判定
    for frame in range(num_frame):  #  各フレームで
        for featurePt_index in range(num_featurePt):  #  各特徴点で
            if np_speed[frame, featurePt_index] < speed_Lh:  #  一定の速さ以下の特徴点を接地していると判定
                grounded_suckers[frame, featurePt_index] = 1  #  接地
            else:
                grounded_suckers[frame, featurePt_index] = 0  #  非接地（プログラム的に冗長だが可読性のために記述）
    
    #  データをpandas.DataFrameの形に変換
    dataframe_grounded_suckers = pd.DataFrame(grounded_suckers)
    dataframe_grounded_suckers.columns = speed.columns
    
    print("DecideGroundedSuckers Done")
    
    return dataframe_grounded_suckers

def DecideGroundedDistalAndProximalArms(
    grounded_suckers: pd.DataFrame,
    sucker_separate: int
):
    """腕の近位と遠位が接地しているか判定

    吸盤の接地判定をもとに腕の近位と腕の遠位が接地しているか判定

    Parameters
    ----------
    grounded_suckers : pd.DataFrame
        吸盤の接地判定
    sucker_separate : int
        根本の吸盤からいくつまで吸盤まで腕の近位とするか

    Returns
    -------
    dataframe_distalProximal_groundedArm: pandas.Dataframe
        腕の近位と遠位が接地しているかの判定
    """
    
    print("DecideGroundedDistalAndProximalArms Start")
    
    num_frame = grounded_suckers.shape[0]  #  総フレーム数
    useSuckers_parts = tool.SeparateIntoParts(grounded_suckers)  #  特徴点のデータを腕ごとにまとめる
    arms = ["r1", "r2", "r3", "r4", "l1", "l2", "l3", "l4"]

    distalProximal_groundedArm = {}  #  結果の保存先を用意
    for arm in arms:  #  各腕で
        
        #  遠位と近位の腕の結果の保存先をそれぞれ用意
        proximal_arm = f"{arm}_proximal"
        distal_arm   = f"{arm}_distal"
        distalProximal_groundedArm[proximal_arm] = np.zeros(num_frame)  #  近位の腕の結果の保存先を用意
        distalProximal_groundedArm[distal_arm]   = np.zeros(num_frame)  #  遠位の腕の結果の保存先を用意

        #  腕の遠位と近位にあたる吸盤をそれぞれ格納
        suckers_proximalArm = useSuckers_parts[arm].iloc[:, 0:sucker_separate]  #  腕の近位にあたる吸盤を格納
        suckers_distalArm   = useSuckers_parts[arm].iloc[:, sucker_separate::]  #  腕の遠位にあたる吸盤を格納
        
        #  腕の遠位と近位の接地判定
        for frame in range(num_frame):  #  各フレームで
            if np.count_nonzero(suckers_proximalArm.iloc[frame, :]) > 1:
                distalProximal_groundedArm[proximal_arm][frame] = 1
            elif np.count_nonzero(suckers_distalArm.iloc[frame, :]) > 1:
                distalProximal_groundedArm[distal_arm][frame] = 1
            
            # if np.count_nonzero(suckers_distalArm.iloc[frame, :]) > 1:
            #     distalProximal_groundedArm[distal_arm][frame] = 1
            # else:
            #     distalProximal_groundedArm[distal_arm][frame] = 0

    dataframe_distalProximal_groundedArm = pd.DataFrame(distalProximal_groundedArm)
    
    print("DecideGroundedDistalAndProximalArms Done")

    return dataframe_distalProximal_groundedArm

def PredictAcceleration(
    acceleration: pd.DataFrame,
    distalProximal_groundedArm: pd.DataFrame,
    stop: int = None
):
    """重回帰分析による加速度の推定とAICによるモデル選択

    Parameters
    ----------
    acceleration : pd.DataFrame
        加速度
    distalProximal_groundedArm : pd.DataFrame
        各腕の近位と遠位の接地判定
    stop : int, optional
        加速度の予測を終了するフレーム, by default None
    Returns
    -------
    acceleration_centre : pd.DataFrame[float]
        推定対象の加速度
    acceleration_centre_predict : pd.DataFrame[float]
        推定した加速度
    start : int
        推定の開始フレーム
    stop : int
        推定のフレーム
    min_grounded_armParts: list[str]
        推定に使用した腕の部位
    coef : list[float]
        各腕の部位の偏回帰係数
    """
    
    print("PredictAcceleration Start")
    
    #  接地する直後と地面から離れるときの推進力への寄与は少ないという仮説をもとに腕の部位の接地判定の移動平均をとる
    filtered_groundedArm = distalProximal_groundedArm.rolling(window=7, center=True).mean()  #  移動平均
    filtered_groundedArm = filtered_groundedArm.drop([0])
    
    acceleration_centre = acceleration.loc[:, "centre"]  #  "centre"の加速度
    
    #  予測を行うフレームの区間の決定
    for frame, acceleration_frame in enumerate(acceleration.loc[:, "mantle"]):  #  "mantle"が写った時点からの加速度の予測を行う
        if ~np.isnan(acceleration_frame):
            start = frame
            break
    if stop != None:  #  加速度の予測を終了するフレームを決定
        filtered_groundedArm = filtered_groundedArm.iloc[start:stop, :]
        acceleration_centre = acceleration_centre[start:stop]
    
    #  重回帰分析による加速度の推定とAICによるモデル選択
    min_aic = np.nan
    num_featurePt = int(filtered_groundedArm.shape[1])  #  腕の部位の数
    for num_usefeature in range(1, num_featurePt+1):  #  加速度の推定に使う腕の部位の数
        for usefeatures in itertools.combinations(filtered_groundedArm.columns, num_usefeature):  #  すべての腕の部位の組み合わせで
            usefeatures = list(usefeatures)  #  推定に使用する腕の部位
            #  加速度の推定
            liner_model = LinearRegression(fit_intercept=False)  #  切片は0とする
            liner_model.fit(filtered_groundedArm[usefeatures], acceleration_centre)
            acceleration_centre_predict = liner_model.predict(filtered_groundedArm[usefeatures])
            
            #  AICの計算
            mse = mean_squared_error(acceleration_centre, acceleration_centre_predict, squared=True)  #  平方二乗誤差の計算
            aic = -2 * len(acceleration_centre) * np.log(1/np.sqrt(2 * np.pi * mse)) + len(acceleration_centre) + num_usefeature  #  AICの計算
            
            if np.isnan(min_aic) or aic < min_aic:  #  AICの値が最小の組み合わせとその時の偏回帰係数とR^2値を保存
                min_aic = aic
                min_grounded_armParts = list(filtered_groundedArm[usefeatures].columns)  #  推定に用いた腕部位
                coef = liner_model.coef_  #  偏回帰係数
                r2 = liner_model.score(filtered_groundedArm[usefeatures], acceleration_centre)  #  R^2値の計算

    print("AIC:", aic)
    print("R^2:", r2)
    
    print("PredictAcceleration Done")
            
    return acceleration_centre, acceleration_centre_predict, start, stop, min_grounded_armParts, coef