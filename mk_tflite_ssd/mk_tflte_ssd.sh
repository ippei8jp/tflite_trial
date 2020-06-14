#!/bin/bash

#### ディレクトリ #############################################################
# tensorflow リポジトリのあるディレクトリ
# tensorflow v2.xだと tensorflow.tools.graph_transforms がないと怒られる... 
TF_BASE_DIR=/work/tf_1.15

# モデルデータのあるディレクトリ
INPUT_MODELS_BASE_DIR=/work/tflite_trial/download_models/_ssd_data
# frozen model を出力するディレクトリ
OUTPUT_FLOZEN_BASE_DIR=`pwd`/_frozen
# tfliteファイルを出力するディレクトリ
OUTPUT_TFLITE_BASE_DIR=`pwd`/_tflite

#### スクリプト実行用ディレクトリに移動 ####
# function で pushd を定義していた場合を回避するため、command をつける
command pushd  ${TF_BASE_DIR}/models/research || {
	# 移動先ディレクトリがなければエラー終了
	echo "ERROR: ######## 指定された models ディレクトリがありません ########"
	exit
}
###############################################################################

#### 各モデル名称と各モデルに対する設定値 #####################################
# 2次元配列が使えないので苦肉の策...
# 以下の配列は同時にセットすること。そうしないと読み出し時に矛盾が起こる。
declare -a names=()               # モデル名
declare -a checkpoints=()         # checkpoint file の prefix
declare -a input_shapeses=()      # 各モデルの input_shapes の値

# input_shapeses の各サイズは 各モデルの pipeline.config の model/ssd/image_resizer/fixed_shape_resizer の height と width から設定する
# tocoのオプションから省略しても大丈夫な気がするが...
names+=('ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssd_mobilenet_v1_coco_2018_01_28')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')
checkpoints+=('model.ckpt')
input_shapeses+=('1,640,640,3')

names+=('ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssd_mobilenet_v2_coco_2018_03_29')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_06')
checkpoints+=('model.ckpt')
input_shapeses+=('1,320,320,3')

names+=('ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssd_mobilenet_v3_large_coco_2020_01_14')
checkpoints+=('model.ckpt')
input_shapeses+=('1,320,320,3')

names+=('ssd_mobilenet_v3_small_coco_2020_01_14')
checkpoints+=('model.ckpt')
input_shapeses+=('1,320,320,3')

names+=('ssdlite_mobilenet_v2_coco_2018_05_09')
checkpoints+=('model.ckpt')
input_shapeses+=('1,300,300,3')

names+=('ssdlite_mobiledet_cpu_320x320_coco_2020_05_19')
checkpoints+=('model.ckpt-400000')
input_shapeses+=('1,320,320,3')

# 以下のパラメータは export_tflite_ssd_graph.py によって指定される固定値
input_arrays='normalized_input_image_tensor'
output_arrays='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3'
###############################################################################

# PYTHONPATHの設定
# 一部ファイルの import に slim が残っているのでslimも追加しておく
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# 各モデルを処理するループ
for ix in ${!names[@]}
do
	model_name=${names[ix]}
	checkpoint=${checkpoints[ix]}
    input_shapes=${input_shapeses[ix]}
	
	# 入出力ファイル/ディレクトリ
	CONFIG_FILE=${INPUT_MODELS_BASE_DIR}/${model_name}/pipeline.config
	CHECKPOINT_PREFIX=${INPUT_MODELS_BASE_DIR}/${model_name}/${checkpoint}
	OUTPUT_FLOZEN_DIR=${OUTPUT_FLOZEN_BASE_DIR}/${model_name}
	OUTPUT_TFLITE_DIR=${OUTPUT_TFLITE_BASE_DIR}/${model_name}
	
	printf "==== %s ====\n" ${model_name}
	
	# 念のため出力ディレクトリを作成しておく
	mkdir -p ${OUTPUT_FLOZEN_DIR}
	mkdir -p ${OUTPUT_TFLITE_DIR}
	
	echo "---- export_tflite_ssd_graph.py ----"
	python object_detection/export_tflite_ssd_graph.py \
	--pipeline_config_path=${CONFIG_FILE} \
	--trained_checkpoint_prefix=${CHECKPOINT_PREFIX} \
	--output_directory=${OUTPUT_FLOZEN_DIR} \
	--add_postprocessing_op=true
	
	# 量子化モデルはfull-integerモデルに変換する
	# NOTE: 
	#       --default_ranges_min=0 --default_ranges_max=255 
	#       を指定すれば量子化モデルでなくても
	#       QUANTIZED_UINT8 に変換できるけど、認識結果が変...
	#       なので、現状はスキップ
	if [[ ${model_name} =~ "quantized" ]]; then
		echo "---- toco ----"
		${TF_BASE_DIR}/tensorflow/bazel-bin/tensorflow/lite/toco/toco \
		--input_file=${OUTPUT_FLOZEN_DIR}/tflite_graph.pb \
		--output_file=${OUTPUT_TFLITE_DIR}/${model_name}.tflite \
		--input_shapes=${input_shapes} \
		--input_arrays=${input_arrays} \
		--output_arrays=${output_arrays} \
		--inference_type=QUANTIZED_UINT8 \
		--mean_values=128 \
		--std_values=128 \
		--change_concat_input_ranges=false \
		--allow_custom_ops
		
		echo "---- edgetpu_compiler ----"
		edgetpu_compiler \
		  --out_dir=${OUTPUT_TFLITE_DIR} \
		   ${OUTPUT_TFLITE_DIR}/${model_name}.tflite
		 
		echo "---- rename cpu qint model ----"
		mv ${OUTPUT_TFLITE_DIR}/${model_name}.tflite ${OUTPUT_TFLITE_DIR}/${model_name}_cpu_qint.tflite
	fi
	
	echo "---- toco(float) ----"
	${TF_BASE_DIR}/tensorflow/bazel-bin/tensorflow/lite/toco/toco \
	--input_file=${OUTPUT_FLOZEN_DIR}/tflite_graph.pb \
	--output_file=${OUTPUT_TFLITE_DIR}/${model_name}_cpu_float.tflite \
	--input_shapes=${input_shapes} \
	--input_arrays=${input_arrays} \
	--output_arrays=${output_arrays} \
	--change_concat_input_ranges=false \
	--allow_custom_ops
done
