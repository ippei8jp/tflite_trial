# ``download_models``ディレクトリにダウンロードしたSSDモデルファイルをtflite形式に変換する

``download_models``ディレクトリにダウンロードしたSSDモデルファイルをtflite形式に変換する。  
量子化モデルはFull integer 量子化を行い、同時にEdge-TPU用tfliteも作成する。  

SSDモデルはそのまま変換しても動かないので、 tensorflow/models リポジトリにある  
``research/object_detection/export_tflite_ssd_graph.py``を使用して変換する。  

# 事前準備

## python環境

python3(動作確認したのは 3.7.7)/tensorflow1.15がインストールされている環境が必要なのでインストールしておく。  
参考：[Google Coral USB Accelerator を使う その5](https://ippei8jp.github.io/memoBlog/2020/05/27/coral_5.html)

その他、以下のコマンドでtf-slimもインストールしておくこと。  
```bash
pip install tf-slim
```

これを含め、動作確認した際の モジュールとバージョンは [requirements.txt](requirements.txt) を参照。

## tensorflow/models

github から tensorflow/models を入手して TF_BASE_DIR に配置しておくこと  
```bash
git clone https://github.com/tensorflow/models.git
```
- 使用した tensorflow/models の git ハッシュ値は 23c87aaa5a1d930f4f9ca927c8c3577056c4f656
- タグを振られたバージョンは research ディレクトリが削除されているので注意すること

## toco 

tflite形式への変換にはtocoが必要なのでbuildしておく。  
tensorflowモジュールをインストールした際に同時にインストールされるtocoとは別物なので注意！！  

参考：[Google Coral USB Accelerator を使う その5](https://ippei8jp.github.io/memoBlog/2020/05/27/coral_5.html)

## edgetpu_compiler

以下を参考に edgetpu_compiler をインストールしておく  

参考：[Google Coral USB Accelerator を使う その2](https://ippei8jp.github.io/memoBlog/2020/05/16/coral_2.html)

# tfliteファイルへの変換

モデルファイルは あらかじめ ``download_models`` ディレクトリでダウンロードしておく。  
``mk_tflte_ssd.sh`` を実行すると、``_frozen`` ディレクトリに frozen model をexportし、  
さらに ``_tflite`` ディレクトリにtfliteファイルを出力する。  

最終的に生成されるファイルは以下の通り。  
| ファイル名                                         | 内容                                              |
|----------------------------------------------------|---------------------------------------------------|
| _ tflite/«モデル名»/«モデル名»_cpu_float.tflite    | frozen model をそのままtfliteに変換したもの       |
| _tflite/«モデル名»/«モデル名»_cpu_qint.tflite      | full integer quantized モデル(量子化モデルのみ)   |
| _tflite/«モデル名»/«モデル名»_cpu_edgetpu.tflite   | Edge-TPU用モデル(量子化モデルのみ)                |

# 参考  

[Running on mobile with TensorFlow Lite](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)  

