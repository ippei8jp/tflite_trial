# SSD実行

## 事前準備
[Google Coral USB Accelerator を使う その1](https://ippei8jp.github.io/memoBlog/2020/05/15/coral_1.html)
 [Google Coral USB Accelerator を使う その2](https://ippei8jp.github.io/memoBlog/2020/05/16/coral_2.html)
を参考にインストールしておく。  

動作確認した際の モジュールとバージョンは [requirements.txt](requirements.txt) を参照。  

## ファイル構成

| ファイル                     | 内容                      |
|------------------------------|---------------------------|
| object_detection_demo_ssd.py | SSD処理スクリプト本体     |
| test.sh                      | テストスクリプト          |
| _result                      | 結果格納用ディレクトリ    |
| requirements.txt             | 使用したpipモジュール一覧 |

## ``object_detection_demo_ssd.py``

SSD認識処理本体。  

USAGEは以下の通り。  

```
usage: object_detection_demo_ssd.py [-h] -m MODEL -i INPUT [-l LABELS]
                                    [--save SAVE] [--time TIME] [--log LOG]
                                    [--no_disp]

optional arguments:
  --save SAVE           Optional. Save result to specified file
  --time TIME           Optional. Save time log to specified file
  --log LOG             Optional. Save console log to specified file
  --no_disp             Optional. without image display

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to a image/video file. (Specify 'cam'
                        to work with camera)
  -l LABELS, --labels LABELS
                        Optional. Labels mapping file
```

モデルファイルを解析し、Edge-TPUモデルであればEdge-TPU用初期化を行う。  
floatモデルの場合、画像データは-1～1の値に正規化して入力している。  



## ``test.sh``

``test.sh`` を実行するとパラメータに応じた設定で ``object_detection_demo_ssd.py`` を実行する。  
オプション/パラメータは以下の通り。

```
  ./test.sh [option_model] [option_log] [model_number | list | all | allall ] [input_file]
    ---- option_model ----
      -c : CPU用量子化モデルを使用
      -t : Edge-TPU用モデルを使用
      -f : CPU用浮動小数点モデルを使用
    ---- option_log ----
      -l : 実行ログを保存(model_number指定時のみ有効
                          all/allall指定時は指定の有無に関わらずログを保存)

    listを指定すると «model_number»とモデル名の対応を表示する
    «model_number»を指定すると指定したモデルファイルで実行する
    allを指定すると、«option_model»の 登録されたすべてのモデルで認識処理を行い、
                     そのログを保存する(その場での表示は行わない)。
    allallを指定すると、CPU用量子化モデル/Edge-TPU用モデル/CPU用浮動小数点モデルの 
                        登録されたすべてのモデルで認識処理を行い、
                        そのログを保存する(その場での表示は行わない)。

    input_file 省略時はデフォルトの入力ファイル(スクリプト内のINPUT_DEFAULT_FILEで指定)を使用
```

ログを保存する場合は、``_result`` ディレクトリに以下の形式で保存される。

| ファイル            | 内容                                            |
|---------------------|-------------------------------------------------|
| «モデル名».log      | 認識結果                                        |
| «モデル名».time     | フレーム毎の処理時間                            |
| «モデル名».«拡張子» | 認識結果画像(入力ファイルと同じフォーマット)    |

### 注意事項  
- ログファイル名はモデル名に応じて付与されるので、同じモデルで入力ファイルを変えて実行すると上書きされる。  
- その場で表示する(ログ保存しない)場合は、認識時間がオリジナルのフレームレートより早い場合、  
  オリジナルのフレームレートに合わせるようにwatが入る。
  最速の認識時間が知りたい場合はｰlオプションでログ保存する。  
  






