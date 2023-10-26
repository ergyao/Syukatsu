# 複数台カメラ撮影用プログラム

複数台のUSB カメラで同期撮影を行うためのプログラム (Windows・Macでの動作は未確認)  
3台カメラで60 fps までほぼ同期（0.01 s単位まで確認）して撮影可能です。 

### テスト環境

ハードウェア
- `CPU`　11th Gen Intel® Core™ i7-11700 @ 2.50GHz × 16
- `GPU`　GeForce GT 1030
- `RAM`　DDR4-2666 (32GiB x 2)
- `Camera`　DMK 33UX273, IMAGINGSOURCE, 3台

ソフトウェア
- `OS`　Ubuntu 22.04.2 LTS
- `g++` (11.3.0)
- `cmake` (3.22.1)
- `opencv` (4.5.4)
- `jsoncpp` (1.9.5)
  
## 実行環境の構築
`jsoncpp` は `sudo apt-get install libjsoncpp-dev` でインストール可能（Linux）

本プログラムのダウンロード
```bash
git clone git@github.com:bcl-group/MultiCameraCapture.git
```
プログラムのコンパイル
```bash
cd MultiCamera
mkdir build
cd build
cmake ..
make
```

## 使い方

カメラの設定ファイル `camera_setting.json` の設定
```json
{
    "cam_num" : 3, // カメラ台数
    "time" : 20, // 撮影時間
    "cam_id" : [0, 2, 4], // カメラID
    "fps" : 60, // フレームレート
    "width" : 640, // 幅
    "height" : 480 // 高さ
}
```
(注意) 上記のJsonファイルを使う場合はコメント文を削除すること

カメラIDは以下のコマンドで確認できる
```bash
v4l2-ctl --list-devices
```
フレームレート・幅・高さは以下のコマンドで確認できる
```bash
v4l2-ctl --list-formats-ext
```
`v4l2-ctl` は以下のコマンドでインストール可能（Linux）
```
sudo apt-get install v4l-utils
```

`camera_setting.json`のパスをコマンドライン引数に指定し実行
```bash
$ ./main ../camera_setting.json # ./main "camera_setting.json"のパス
```

## 問題点
- 120 fps に設定すると周期的にコマ落ちが発生する
- imshow 関数の処理速度が遅く、3台表示させるとフレーム間隔が 1 ms ばらつく (表示をなくすと安定)
## 改善案
- 処理の重いimshow を openGL の関数へ置き換える
- スレッドを使いカメラの撮影を並列化を行う
- 関数を作ってmain 文を簡素化を行う