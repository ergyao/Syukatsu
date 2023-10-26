from preprocessing import Preprocess
from analyze import Analyze
from plot import CoordinatesPlot
from plot import ResultPlot

class AnalyzeCrawling:
    
    def __init__(self):
        """パラメータの設定と前処理の実行
        
        パラメータをデータに合わせた値に変更してください。
        
        """
    
        """
        データの読み込み
        """
        path_csv = "data/coordinates/f02_s1_cam02_1213_free1.csv"  #  "DeepLabCut"で得られた特徴点座標のcsvファイル (str)
        path_cal = "data/calibration/cal.pkl"  #  "calibration/mark_calibration.py"でキャリブレーションを行ったときの出力ファイルである"cal.pkl" (str)

        """
        前処理
        """
        Th_Lh = 0.8  #  欠損値のしきい値 (float, 0~1)
        Th_data = 0.3  #  データ数のしきい値（"centre"の何割か）(float, 0~1)
        self.fs = 30  #  フレームレート (int)
        self.fc = 3  #  ローパスフィルターのカットオフ周波数 (int)

        #  解析データの保存
        self.save = False  # 解析データを逐一保存するか (bool)
        self.path_save = "result"  #  解析データを保存する場所 (str)

        """
        解析
        """
        #  接地判定
        self.speed_Lh = 10  #  接地と判定する速さのしきい値(mm)(float)
        #  腕の近位と遠位の接地判定
        self.sucker_separate = 5  #  根本の吸盤からいくつまで吸盤まで腕の近位とするか (int)
        #  重回帰分析による加速度の推定とAICによるモデル選択
        self.stop = 160 #  加速度の予測をどのフレームまで行うか(int)


        """
        前処理の実行
        """
        self.coordinates = Preprocess.Preprocess(path_csv=path_csv, path_cal=path_cal, Th_Lh=Th_Lh, Th_data=Th_data, fs=self.fs, fc=self.fc, path_save=self.path_save, save=self.save)
        
    def GroundedSuckers(self, arm:str):
        try:
            self.velocity
        except AttributeError:
            self.velocity = Analyze.CalculateVelocity(coordinates=self.coordinates, fs=self.fs, fc=self.fc)
        
        try:
            self.speed
        except AttributeError:
            self.speed = Analyze.CalculateSpeed(velocity=self.velocity)
            
        try:
            self.grounded_suckers
        except AttributeError:
            self.grounded_suckers = Analyze.DecideGroundedSuckers(self.speed, self.speed_Lh)
        
        ResultPlot.PlotGroundedSuckers(grounded_suckers=self.grounded_suckers, arm=arm, fs=self.fs)
    
    def ArmCurve(self, arm:str):
        try:
            self.arm_curve
        except AttributeError:
            self.arm_curve = Analyze.EvaluateArmCurve(coordinates=self.coordinates)
        
        ResultPlot.PlotArmCurve(arm_curve=self.arm_curve, arm=arm, fs=self.fs)
        
    def DistanceFromCentre(self, arm:str, index_sucker:int):
        try:
            self.distance_fromCentre
        except AttributeError:
            self.distance_fromCentre = Analyze.CaluculateDistanceFromCentre(coordinates=self.coordinates)
        
        ResultPlot.PlotDistanceFromCentre(distance_fromCentre=self.distance_fromCentre, arm=arm, index_sucker=index_sucker, fs=self.fs)
        
    def AccelerationPredict(self, stop:int):
        try:
            self.velocity
        except AttributeError:
            self.velocity = Analyze.CalculateVelocity(coordinates=self.coordinates, fs=self.fs, fc=self.fc)
        
        try:
            self.speed
        except AttributeError:
            self.speed = Analyze.CalculateSpeed(velocity=self.velocity)
        
        try:
            self.acceleration
            self.acceleration_xy
        except AttributeError:
            self.acceleration, self.acceleration_xy = Analyze.CalculateAcceleration(velocity=self.velocity, speed=self.speed, fs=self.fs)
        
        try:
            self.grounded_suckers
        except AttributeError:
            self.grounded_suckers = Analyze.DecideGroundedSuckers(self.speed, self.speed_Lh)
        
        try:
            self.distalProximal_groundedArm
        except AttributeError:
            self.distalProximal_groundedArm = Analyze.DecideGroundedDistalAndProximalArms(grounded_suckers=self.grounded_suckers,sucker_separate=self.sucker_separate)
        
        try:
            self.acceleration_centre
            self.acceleration_centre_predict
            self.start
            self.stop
            self.min_grounded_armParts
            self.coefficient
        except AttributeError:
            self.acceleration_centre, self.acceleration_centre_predict, self.start, self.stop, self.min_grounded_armParts, self.coefficient = Analyze.PredictAcceleration(acceleration=self.acceleration, distalProximal_groundedArm=self.distalProximal_groundedArm, stop=stop)
        
        ResultPlot.PlotAccelerationPredict(acceleration_target=self.acceleration_centre, acceleration_predict=self.acceleration_centre_predict, start=self.start, stop=self.stop, fs=self.fs, num_frame=self.acceleration.shape[0])
        ResultPlot.PlotCoefficient(grounded_armParts=self.min_grounded_armParts, coefficient=self.coefficient)
        
    def CheckCoordinate(self, frame_first:int=0, frame_end:int=200, interval:int=100, fileName_save: str = "coordinates.gif"):
        CoordinatesPlot.PlotAllCoordinates(coordinates=self.coordinates, frame_first=frame_first, frame_end=frame_end, interval=interval, save=self.save, path_save=self.path_save, fileName_save=fileName_save)