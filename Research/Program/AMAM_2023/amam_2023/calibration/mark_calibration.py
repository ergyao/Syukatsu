import numpy as np
import pandas as pd
import cv2
import tkinter as tk

def cal(
    cal_picture_path: str = "label.BMP",
    n_row: int = 3,
    n_column: int = 3,
    store: str = "cal.pkl"
):
    '''キャリブレーションのラベル付け

    Parameters:
    ----------
    cal_picture_path:str
        キャリブレーションのpath
    n_row:int
        キャリブレーションの行の数
    n_column:int
        キャリブレーションの列の数
    store:str
        保存先のパス

    '''
    def onMouse(event, x, y, flags, params):
        nonlocal img
        nonlocal n_img
        nonlocal p
        nonlocal n_point

        if event == cv2.EVENT_LBUTTONDOWN and p < n_point:

            print(f"point{p}")
            print(x, y)
            np.put(tx_label, p, x)
            np.put(ty_label, p, y)
            cv2.circle(
                img,  # 図形入力画像
                (x, y),  # 開始点の座標(X,Y)
                1,  # 半径
                (0, 0, 255),  # カラーチャネル(B,G,R)
                -1  # セルの太さ（-1にすると塗りつぶしになる）
            )
            p += 1
            if p == n_point:
                print("全てのポイントをマークしました")

        if event == cv2.EVENT_MBUTTONDOWN and p < n_point:
            print(f"point{p} skip")
            p += 1
            if p == n_point:
                print("全てのポイントをマークしました")

        if event == cv2.EVENT_RBUTTONDOWN and p > 0:

            p -= 1
            np.put(tx_label, p, np.nan)
            np.put(ty_label, p, np.nan)

            img = n_img.copy()
            print(tx_label[~np.isnan(tx_label)])
            for tx, ty in zip(
                    tx_label[~np.isnan(tx_label)].astype('int64'),
                    ty_label[~np.isnan(ty_label)].astype('int64')
                    ):
                cv2.circle(
                    img,  # 図形入力画像
                    (tx, ty),  # 開始点の座標(X,Y)
                    1,  # 半径
                    (0, 0, 255),  # カラーチャネル(B,G,R)
                    -1  # セルの太さ（-1にすると塗りつぶしになる）
                )

            print(f"point{p} delete")

    n_point = n_row * n_column
    x_label = np.empty((0, n_point))
    y_label = np.empty((0, n_point))
    tx_label = np.full(n_point, np.nan)
    ty_label = np.full(n_point, np.nan)
    p = 0
    img = cv2.imread(cal_picture_path)
    n_img = img.copy()
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", onMouse)
    while (True):
        cv2.imshow("label", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    del img
    print(tx_label)
    print(ty_label)

    x_label = np.vstack([x_label, tx_label])
    y_label = np.vstack([y_label, ty_label])

    print(x_label)
    print(y_label)
    p_index = []
    for i in range(n_point):
        p_index.append(f"point{i}")

    label = pd.DataFrame(
        data={
            'x': np.nanmean(x_label, axis=0),
            'y': np.nanmean(y_label, axis=0)},
        index=p_index
        )
    print(label)
    label.to_pickle(store)
