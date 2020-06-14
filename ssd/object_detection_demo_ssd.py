#!/usr/bin/env python
import sys
import platform
import os
import time
import math
import logging as log
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np

import tflite_runtime.interpreter as tflite

# shared library
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def console_print(log_f, message, both=False) :
    if not (log_f and (not both)) :
        print(message)
    if log_f :
        log_f.write(message + '\n')

# コマンドラインパーサの構築
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a image/video file. (Specify 'cam' to work with camera)", required=True, type=str)
    args.add_argument("-l", "--labels", help="Optional. Labels mapping file", default=None, type=str)
    parser.add_argument("--save", help="Optional. Save result to specified file", default=None, type=str)
    parser.add_argument("--time", help="Optional. Save time log to specified file", default=None, type=str)
    parser.add_argument("--log",  help="Optional. Save console log to specified file", default=None, type=str)
    parser.add_argument("--no_disp", action='store_true', help="Optional. without image display")
    return parser

# interpreter の生成
def make_interpreter(model_file, log_f):
    # CPU/TPU使用の識別
    # 「ファイル名に"_edgetpu"が含まれていたら」の識別方法もアリかもしれない
    with open(model_file, "rb") as f:
        # モデルデータを読み込む
        tfdata = f.read()
        # モデルファイル中に"edgetpu-custom-op"が含まれていたらTPU使用モデル
        cpu = not b"edgetpu-custom-op" in tfdata
    
    if cpu :
        console_print(log_f, '**** USE CPU ONLY!! ****', True)
    else :
        console_print(log_f, '**** USE WITH TPU ****', True)

    if cpu :
        return tflite.Interpreter(model_path=model_file)
    else :
        return tflite.Interpreter(
                model_path = model_file,
                experimental_delegates = [
                    tflite.load_delegate(EDGETPU_SHARED_LIB)
                ])

# 結果の解析と表示
def parse_result(interpreter, frame, labels_map, args, log_f) :
    
    img_height, img_width = frame.shape[:2]
    output_details = interpreter.get_output_details()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    tflite_results1 = interpreter.get_tensor(output_details[0]['index'])  # Locations (Top, Left, Bottom, Right)
    tflite_results2 = interpreter.get_tensor(output_details[1]['index'])  # Classes (0=Person)
    tflite_results3 = interpreter.get_tensor(output_details[2]['index'])  # Scores
    tflite_results4 = interpreter.get_tensor(output_details[3]['index'])  # Number of detections
    
    for i in range(int(tflite_results4[0])):
        top    = tflite_results1[0, i][0] * input_height
        left   = tflite_results1[0, i][1] * input_width
        bottom = tflite_results1[0, i][2] * input_height
        right  = tflite_results1[0, i][3] * input_width
        
        # ラベルが定義されていればラベルを読み出し、なければclass ID
        class_id = tflite_results2[0, i].astype(int) + 1
        if labels_map :
            if len(labels_map) > class_id :
                class_name = labels_map[class_id]
            else :
                class_name = str(class_id)
        else :
            class_name = str(class_id)

        prob = tflite_results3[0, i]
        if prob >= 0.5:
            console_print(log_f, f'Class={class_name:15}    Probability={prob:4f}    Location=({int(left)},{int(top)})-({int(right)},{int(bottom)})', False)
            top    = int(top    * img_height  / input_height)
            left   = int(left   * img_width / input_width)
            bottom = int(bottom * img_height  / input_height)
            right  = int(right  * img_width / input_width)
            # 表示色
            if class_id == 1 :
                # class_id = 1のときは緑
                color = (128, 255, 128)
            else :
                # それ以外のときは赤
                color = (128, 128, 255)
            # 対象物の枠とラベルの描画
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top+20), (left+160, top), color, -1)
            cv2.putText(frame, "{} ({:.3f})".format(class_name, prob),
                        (left, top+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    return

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    
    model = args.model
    no_disp = args.no_disp
    
    model_label = None
    if args.labels:
        model_label = args.labels
    else:
        model_label = os.path.splitext(model)[0] + ".labels"
    if not os.path.isfile(model_label)  :
        model_label = None
    
    # ログファイル類の初期化
    time_f = None
    if args.time :
        time_f = open(args.time, mode='w')
        time_f.write(f'frame_number, frame_time, preprocess_time, inf_time, parse_time, render_time, wait_time\n')     # 見出し行
    
    log_f = None
    if args.log :
        log_f = open(args.log, mode='w')
        log_f.write(f'command: {" ".join(sys.argv)}\n')     # 見出し行
    
    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    
    # interpreterの構築
    interpreter = make_interpreter(args.model, log_f)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    
    # 入力データ
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = os.path.abspath(args.input)
        assert os.path.isfile(input_stream), "Specified input file doesn't exist"
    
    cap = cv2.VideoCapture(input_stream)
    
    # 幅と高さを取得
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 90      # 情報表示領域分を追加
    # フレームレート(1フレームの時間単位はミリ秒)の取得
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))                 # オリジナルのフレームレート
    org_frame_time = 1.0 / cap.get(cv2.CAP_PROP_FPS)                # オリジナルのフレーム時間
    
    if model_label:
        # ラベルファイルの読み込み
        with open(model_label, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        # ラベルファイルがない場合
        labels_map = None
    
    # フレーム数
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = 1 if all_frames != -1 and all_frames < 0 else all_frames
    
    writer = None
    jpeg_file = None
    if args.save :
        if all_frames == 1 :
            jpeg_file = args.save
        else :
            # フォーマット
            fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(args.save, fmt, org_frame_rate, (img_width, disp_height))
    
    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    
    # 実行時間測定用変数の初期化
    frame_time = 0
    preprocess_time = 0
    inf_time = 0
    parse_time = 0
    render_time = 0

    # フレーム測定用タイマ
    prev_time = time.time()
    frame_number = 1
    
    # モデルの入力サイズ
    _, input_height, input_width, _ = input_details[0]['shape']
    
    # モデルの入力データ型
    input_dtype = input_details[0]['dtype']
    # print(f'input_dtype = {input_dtype}')
    
    while cap.isOpened():
        # 画像の前処理 =============================================================================
        preprocess_start = time.time()                          # 前処理開始時刻
        
        # 画像の読み込み
        ret, frame = cap.read()
        if not ret:
            break
        
        # 現在のフレーム番号表示
        console_print(log_f, f'frame_number: {frame_number:5d} / {all_frames}', True)
        
        # 表示用領域を含んだフレームを作成
        pad_img = np.zeros((disp_height, img_width, 3), np.uint8)
        
        # モデル入力用にリサイズ
        inf_start = time.time()                                 # 推論処理開始時刻
        in_frame = cv2.resize(frame, (input_width, input_height))   # input size of coco ssd mobilenet?
        in_frame = in_frame[:, :, [2,1,0]]                          # BGR -> RGB
        in_frame = np.expand_dims(in_frame, axis=0)                 # 3D -> 4D
        if input_dtype != np.uint8 :
            in_frame = in_frame.astype(input_dtype)                 # 入力型がuint8以外だったら型変換
            in_frame = (in_frame -128) / 128                        # 値を -1 ～ 1 の範囲に正規化
        
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start     # 前処理にかかった時間
        
        # 推論実行 =============================================================================
        inf_start = time.time()                                 # 推論処理開始時刻
        # 推論本体
        interpreter.set_tensor(input_details[0]['index'], in_frame)
        interpreter.invoke()
        
        inf_end = time.time()
        inf_time = inf_end - inf_start                          # 推論処理にかかった時間
        
        # 検出結果の解析 =============================================================================
        parse_start = time.time()                               # 解析処理開始時刻
        parse_result(interpreter, frame, labels_map, args, log_f)
        parse_end = time.time()
        parse_time = parse_end - parse_start                    # 解析処理にかかった時間
        
        # 結果の表示 =============================================================================
        render_start = time.time()                              # 表示処理開始時刻
        
        # 測定データの表示
        frame_number_message    = f'frame_number   : {frame_number:5d} / {all_frames}'
        if frame_time == 0 :
            frame_time_message  =  'Frame time     : ---'
        else :
            frame_time_message  = f'Frame time     : {(frame_time * 1000):.3f} ms    {(1/frame_time):.2f} fps'
        inf_time_message        = f'Inference time : {(inf_time * 1000):.3f} ms'
        render_time_message     = f'Rendering time : {(render_time * 1000):.3f} ms'
        parsing_time_message    = f'parse time     : {(parse_time * 1000):.3f} ms'
        
        # 結果の書き込み
        cv2.putText(pad_img, frame_number_message, (10, img_height + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(pad_img, inf_time_message,     (10, img_height + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(pad_img, parsing_time_message, (10, img_height + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(pad_img, render_time_message,  (10, img_height + 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
        cv2.putText(pad_img, frame_time_message,   (10, img_height + 75), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
        
        # 表示用領域に画像をコピー
        pad_img[:img_height, :img_width] = frame
        
        # 表示
        if not no_disp :
            cv2.imshow("Detection Results", pad_img)        # 表示
        
        # 画像の保存
        if jpeg_file :
            # jpeg 保存モードならjpegで保存
            cv2.imwrite(jpeg_file, pad_img)
        elif writer:
            # 保存が設定されているときは画像を保存
            writer.write(pad_img)
        
        render_end = time.time()
        render_time = render_end - render_start                 # 表示処理にかかった時間
        
        # タイミング調整 =============================================================================
        wait_start = time.time()
        if no_disp :
            # 表示しない場合は無駄な待ち時間を確保しない
            wait_key_code = 1 
        else :
            # フレーム先頭からここまでの時間
            cur_total_time = wait_start - preprocess_start
            # フレーム間処理の待ち時間(フレームが1枚だけの場合は永久待ち)
            if all_frames == 1:
                wait_key_code = 0
            else :
                if org_frame_time < cur_total_time :
                    # オリジナルフレーム時間がここまでの時間より短ければ最短時間
                    wait_key_code = 1 
                else :
                    # オリジナルフレーム時間とここまでの時間の差(msec単位に変換して小数点以下切り上げ)
                    wait_key_code = math.ceil((org_frame_time - cur_total_time) * 1000)
        key = cv2.waitKey(wait_key_code)
        if key == 27:
            # ESCキー
            break
        wait_end = time.time()
        wait_time = wait_end - wait_start
        
        # フレーム処理終了 =============================================================================
        cur_time = time.time()
        frame_time = cur_time - prev_time         # 1フレームの処理時間
        prev_time = cur_time
        if time_f :
            time_f.write(f'{frame_number:5d}, {frame_time * 1000:.3f}, {preprocess_time * 1000:.3f}, {inf_time * 1000:.3f}, {parse_time * 1000:.3f}, {render_time * 1000:.3f}, {wait_key_code}, {wait_time * 1000:.3f}\n')
        frame_number = frame_number + 1
    
    # 後片付け
    if writer:
        writer.release()
    if time_f :
        time_f.close()
    if log_f :
        log_f.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
