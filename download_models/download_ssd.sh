#!/bin/bash

# モデルの格納場所一覧
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

DATA_DIR=_ssd_data

# ダウンロードディレクトリの作成
mkdir -p ${DATA_DIR}
# モデルのダウンロードと展開
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz                                            | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz                  | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz                   | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz        | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz    | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz    | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz                                            | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz                          | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz                                        | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_06.tar.gz  | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz                                      | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz                                      | tar xzvf - -C ${DATA_DIR}
wget -O -  http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz                               | tar xzvf - -C ${DATA_DIR}

# ssd_mobilenet_v2_coco_2018_03_29 はそのままでは以下のエラーになる
#         Message type "object_detection.protos.SsdFeatureExtractor" has no field named "batch_norm_trainable".
# 下記パッチをあてることで回避
# なぜか-lオプションをつけないとファイルが見つからないと怒られる…
pushd ${DATA_DIR}
patch -l -p1 << __EOF__
diff -ur _data.org/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config _data/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
--- _data.org/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config	2018-03-30 11:48:19.085129000 +0900
+++ _data/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config	2020-06-03 07:33:58.842219659 +0900
@@ -32,7 +32,7 @@
           train: true
         }
       }
-      batch_norm_trainable: true
+      # batch_norm_trainable: true
       use_depthwise: true
     }
     box_coder {
__EOF__

# ラベルデータファイルの作成
wget -O - https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_complete_label_map.pbtxt \
	| grep display_name | sed -e "s/^.*\"\(.*\)\".*$/\1/g" >  mscoco_complete_label_map.labels


popd
