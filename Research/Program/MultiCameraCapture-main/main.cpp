#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
/*
    複数台のカメラ撮影
    ・使い方
        コマンドライン引数にカメラ設定用のJsonファイルのパスを指定して実行する
        （例）./main ../camera_setting.json
    ・Json ファイル (JSON の方に書き写す)
        設定項目
        ・カメラの台数 cam_num int型
        ・カメラID cam_id int型の一次元配列
        ・フレームレート fps int型
        ・幅 width int型
        ・高さ height int型
        ・撮影時間 time int型
*/
void check_fps(int num_frame, std::vector<std::chrono::system_clock::time_point> time_stamp, bool interval = false)
{
    double ave; // 平均
    double std; // 標準偏差
    double sum1 = 0.0; // 合計1
    double sum2 = 0.0; // 合計2
    std::vector<int> time_difference(num_frame-1); // フレーム間の時間間隔
    // 平均の計算・フレーム間の時間表示
    for (int i=0; i<num_frame -1; i++)
    {
        auto t = time_stamp[i+1] - time_stamp[i];
        time_difference[i] = std::chrono::duration_cast<std::chrono::milliseconds>(t).count();
        if (interval)
        {
            std::cout << i << ">" << i+1 << ": " << time_difference[i] << " ms" << std::endl;
        }
        sum1 += time_difference[i];
    }
    ave = sum1 / num_frame;
    // 標準偏差の計算
    for (int i=0; i<num_frame -1; i++)
    {
        sum2 += (time_difference[i]-ave)*(time_difference[i]-ave);
    }
    std = sqrt(sum2/num_frame);
    // デバック結果の表示
    std::cout << "FrameRate: " << std::endl;
    std::cout << "AVE: " << ave << std::endl;
    std::cout << "STD: " << std << std::endl;
}

int main(int argc,char* argv[])
{
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    // Jsonファイルの読み込み
    std::ifstream set_file(argv[1]);
    if (!set_file)
    {
        std::cout << "Error :Can't open \"camera_setting.json\"" << std::endl;
        std::cout << "Please, Check \"camera_setting.json\" path in command line arguments" << std::endl;
        std::cout << "(ex. ./main ../camera_setting.json)" << std::endl;
        std::cout << argv << std::endl;
        return 0;
    }
    Json::Reader reader;
    Json::Value cam_prop;
    reader.parse (set_file, cam_prop);
    set_file.close();


    // カメラのプロパティ・ビデオの設定
    // カメラ
    int cam_num = cam_prop["cam_num"].asInt64(); // カメラの台数
    int cam_fps = cam_prop["fps"].asInt64(); // フレームレート 
    int cam_w = cam_prop["width"].asInt64(); // 幅
    int cam_h = cam_prop["height"].asInt64(); // 高さ
    std::vector<int> cam_id(cam_num); // カメラのデバイスID
    for (int i; i<cam_num; i++)
    {
        cam_id[i] = cam_prop["cam_id"][i].asInt64();
    }
    std::vector<cv::VideoCapture> cap(cam_num); // カメラのインスタンス
    std::vector<std::string> cam_name(cam_num); // 表示時のウィンドウ名
    // ビデオ
    int fourcc = cv::VideoWriter::fourcc('M','P','4','V'); // 保存形式
    std::vector<cv::VideoWriter> video(cam_num); // ビデオのインスタンス
    std::string video_name; //　保存ファイル名
    // カメラのインスタンスへの設定
    for (int i; i<cam_num; i++)
    {

        cap[i] = cv::VideoCapture(cam_id[i]);
        cap[i].set(cv::CAP_PROP_FPS, cam_fps);
        cap[i].set(cv::CAP_PROP_FRAME_WIDTH, cam_w);
        cap[i].set(cv::CAP_PROP_FRAME_HEIGHT, cam_h);
        cap[i].set(cv::CAP_PROP_BUFFERSIZE, 1);
        cam_name[i] = "cam" + std::to_string(i);

        video_name = "video" + std::to_string(i) + ".mp4";
        video[i].open(video_name, fourcc, cam_fps, cv::Size(cam_w, cam_h));
    }


    // 計測準備
    int capture_time = cam_prop["time"].asInt64(); // 計測時間
    int total_frame = cam_fps * capture_time; // 必要なフレーム数
    std::vector<std::vector<cv::Mat>> frame(cam_num, std::vector<cv::Mat>(total_frame)); // フレームの一時保存用配列の確保
    std::vector<std::chrono::system_clock::time_point> time_stamp(total_frame); // タイムスタンプ（フレームレート計算用）

    int frame_count = 0; // 撮影したフレーム数
    while (frame_count < total_frame)
    {

        for (int i=0; i<cam_num; i++)
        {
            cap[i].grab(); // 画像の取り込み
        }
        for (int i=0; i<cam_num; i++)
        {
            cap[i].retrieve(frame[i][frame_count]); // 画像のデコード・メモリ（配列）への一時保存
            // cam_name = "cam" + std::to_string(i);
            cv::imshow(cam_name[i], frame[i][frame_count]); // 表示
        }

        time_stamp[frame_count++] = std::chrono::system_clock::now(); // タイムスタンプ記録
        const int key = cv::waitKey(1);
        if(key == 'q') // "q" で強制終了
        {
            break;
        }
    }
    cv::destroyAllWindows();

    // フレームレートの計算（デバック）
    check_fps(frame_count, time_stamp, true);

    // 動画の作成
    std::cout << "Please, wait. Now, making video files." << std::endl;
    for (int i=0; i<cam_num; i++)
    {
        for (int j=0; j<frame_count; j++)
        {
            video[i] << frame[i][j]; // 一時保存したフレームを動画ファイルに書き込み
        }
    }

    return 0;
}