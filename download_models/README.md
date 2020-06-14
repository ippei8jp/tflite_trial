# インターネット上で配布されているモデルファイルをダウンロードする

``download_ssd.sh``を実行すると [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models)
で公開されているSSDのモデルファイルを``_ssd_data``ディレクトリにダウンロードして展開する。  

一部、修正が必要なファイルにはパッチを当てている。  

また、認識結果表示用に使用するラベルデータも作成する。  

