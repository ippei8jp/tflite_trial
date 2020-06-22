#!/bin/bash

# コマンド名
COMMAND_NAME=$0

# 処理本体のファイル名
MAIN_SCRIPT=object_detection_demo_ssd.py

# プロジェクトのベースディレクトリ
BASE_DIR=/work/tflite_trial

# 入出力ファイル/ディレクトリ
LABEL_FILE=${BASE_DIR}/download_models/_ssd_data/mscoco_complete_label_map.labels 
TFLITE_BASE=${BASE_DIR}/mk_tflite_ssd/_tflite/
INPUT_DEFAULT_FILE=${BASE_DIR}/images/testvideo3.mp4
RESULT_DIR=./_result

# 結果格納ディレクトリを作っておく
mkdir -p ${RESULT_DIR}

# モデル名リスト
    # ==== メモ ==== 
    # inline comment は 「`#～`」 で囲むと可能。
    # 行頭の場合はさらに 「;」 を付け加えると安心。

MODEL_NAMES=()   # 初期化
`#  0`;MODEL_NAMES+=("ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03")					# 0.75 は depth_multiplier の値を示している
`#  1`;MODEL_NAMES+=("ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18")
`#  2`;MODEL_NAMES+=("ssd_mobilenet_v1_coco_2018_01_28")											# ssd_mobilenet_v1 のSSDの基本モデル？
# ↓は ものすごく遅い
`#  3`;MODEL_NAMES+=("ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03")	# fpn: Feature Pyramid Net
`#  4`;MODEL_NAMES+=("ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03")	# ppn: Pooling Pyramid Network
`#  5`;MODEL_NAMES+=("ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18")					# ssd_mobilenet_v1 のSSDの量子化モデル？
`#  6`;MODEL_NAMES+=("ssd_mobilenet_v2_coco_2018_03_29")											# ssd_mobilenet_v2 のSSDの基本モデル？
`#  7`;MODEL_NAMES+=("ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_06")	# mnasfpn: ??
`#  8`;MODEL_NAMES+=("ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03")							# ssd_mobilenet_v2 のSSDの量子化モデル？
`#  9`;MODEL_NAMES+=("ssd_mobilenet_v3_large_coco_2020_01_14")										# ssd_mobilenet_v3 のlargeモデル？
`# 10`;MODEL_NAMES+=("ssd_mobilenet_v3_small_coco_2020_01_14")										# ssd_mobilenet_v3 のsmallモデル？
`# 11`;MODEL_NAMES+=("ssdlite_mobilenet_v2_coco_2018_05_09")
`# 12`;MODEL_NAMES+=("ssdlite_mobiledet_cpu_320x320_coco_2020_05_19")

# ======== USAGE 表示 =================================================
usage(){
	echo '==== USAGE ===='
	echo "  ${COMMAND_NAME} [option_model] [option_log] [model_number | list | all | allall ] [input_file]"
	echo '    ---- option_model ----'
	echo '      -c : CPU用量子化モデルを使用'
	echo '      -t : Edge-TPU用モデルを使用'
	echo '      -f : CPU用浮動小数点モデルを使用'
	echo '    ---- option_log ----'
	echo '      -l : 実行ログを保存(model_number指定時のみ有効'
	echo '                          all/allall指定時は指定の有無に関わらずログを保存)'
	echo '    input_file 省略時はデフォルトの入力ファイルを使用'
	echo ' '
}

# ======== USAGE 表示 =================================================
disp_list() {
	# usage 表示
	usage
	echo '==== directories ===='
	echo "TFLITE_BASE : ${TFLITE_BASE}"
	echo ' '
	echo '==== MODEL LIST ===='
	count=0
	for MODEL_NAME in ${MODEL_NAMES[@]}
	do 
		echo "${count} : ${MODEL_NAME}"
		count=`expr ${count} + 1`
	done
	exit
}
# ======== コマンド本体実行 =================================================
execute() {
	# モデルファイル
	local MODEL_FILE=${TFLITE_BASE}/${MODEL_NAME}/${MODEL_NAME}${model_name_ext}.tflite
	if [ ! -f ${MODEL_FILE} ]; then
		# 指定されたモデルファイルが存在しない
		echo "==== モデルファイル ${MODEL_FILE} は存在しません ===="
		return 1
	fi
	if [ "${log_flag}" != "log" ] ; then
		python ${MAIN_SCRIPT} --input ${INPUT_FILE} --label ${LABEL_FILE} --model ${MODEL_FILE}
	else
		# 保存ファイル名など
		local SAVE_EXT=${INPUT_FILE##*.}    # 入力ファイルの拡張子
		local SAVE_NAME=${RESULT_DIR}/${MODEL_NAME}${model_name_ext}.${SAVE_EXT}
		local TIME_FILE=${RESULT_DIR}/${MODEL_NAME}${model_name_ext}.time
		local LOG_FILE=${RESULT_DIR}/${MODEL_NAME}${model_name_ext}.log
		python ${MAIN_SCRIPT} --input ${INPUT_FILE} --label ${LABEL_FILE} --model ${MODEL_FILE} --save ${SAVE_NAME} --time ${TIME_FILE} --no_disp --log ${LOG_FILE}

		# 実行時間の平均値を計算してファイルに出力
		python -c "import sys; import pandas as pd; data = pd.read_csv(sys.argv[1], index_col=0); ave=data.mean(); print(ave)" ${TIME_FILE} > ${TIME_FILE}.average
	fi
	return 0
}

# ======== 自動実行処理 =================================================
all_execute() {
	# 実行時間記録ファイル
	local TIME_LOG=${RESULT_DIR}/time${model_name_ext}.txt
	echo "各モデルの実行時間" > ${TIME_LOG}
	
	# 各モデルに対する処理ループ
	for MODEL_NAME in ${MODEL_NAMES[@]}
	do 
		# 前処理
		echo "######## ${MODEL_NAME}${model_name_ext} ########" | tee -a ${TIME_LOG}
		
		# 実行開始時刻(秒で取得して日付で表示)
		local start_time=`date +%s`
		echo "***START*** : `date -d @${start_time} +'%Y/%m/%d %H:%M:%S'`" | tee -a ${TIME_LOG}
		
		if [[ ${MODEL_NAME} == *ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03* ]];
		then
			echo "***SKIP *** : ${MODEL_NAME} は遅いのでスキップします" | tee -a ${TIME_LOG}
			continue
		fi
		
		# 処理本体
		execute
		local EXEC_RET=$?		# 処理本体の戻り値
		
		# 後処理
		if [ ${EXEC_RET} -eq 0 ]; then
			# 実行終了時刻(秒で取得して日付で表示)
			local end_time=`date +%s`
			echo "*** END *** : `date -d @${end_time} +'%Y/%m/%d %H:%M:%S'`" | tee -a ${TIME_LOG}
			# 実行時間(ちょっと姑息な方法で "秒数" を "時:分:秒" に変換)
			local execution_time=$(expr ${end_time} - ${start_time})
			echo "=== Execution time : `TZ=0 date -d@${execution_time} +%H:%M:%S`" | tee -a ${TIME_LOG}
		else
			echo "***SKIP *** : ${MODEL_NAME} のモデルファイルは存在しません " | tee -a ${TIME_LOG}
		fi
	done
}
# ======== 全自動実行処理 =================================================
allall_execute() {
	for model_name_ext in "_cpu_qint" "_edgetpu" "_cpu_float"
	do
		all_execute
	done
}

# オプション連動変数
model_name_ext="_cpu_float"		# デフォルトはCPU用浮動小数点モデル
log_flag=""

# オプション解析
while getopts :ctfl OPT
do
  case $OPT in
    "c" ) model_name_ext="_cpu_qint" ;;
    "t" ) model_name_ext="_edgetpu" ;;
    "f" ) model_name_ext="_cpu_float" ;;
    "l" ) log_flag="log" ;;
    "?" ) usage; exit ;;
  esac
done

# オプション部分を切り捨てる
shift `expr $OPTIND - 1`

# 引数の個数
num_args=$#

if [ ${num_args} -eq 0 ] ;then
	disp_list			# リスト表示
	exit
fi

if [ ${num_args} -ge 2 ] ;then
	# 第2パラメータがあったら入力ファイルを変更
	INPUT_FILE=$2
else
	INPUT_FILE=${INPUT_DEFAULT_FILE}
fi

if [ "$1" == "list" ] ;then
	disp_list			# リスト表示
	exit
elif [ "$1" == "all" ] ;then
    log_flag="log"		# ログ保存有効
	all_execute			# 自動実行
	exit
elif [ "$1" == "allall" ] ;then
    log_flag="log"		# ログ保存有効
	allall_execute		# 自動実行
	exit
else
	MODEL_NO=$1			# モデル番号
fi

num_models=${#MODEL_NAMES[@]}

# パラメータが数値か確認
expr "${MODEL_NO}" + 1 >/dev/null 2>&1
if [ $? -ge 2 ]
then
  echo "${MODEL_NO}"
  echo " 0 以上 $num_models 未満の数値を指定してください(1)"
  usage
  exit
fi
# パラメータが正数か確認
if [ ${MODEL_NO} -lt 0 ] ; then
  echo " 0 以上 $num_models 未満の数値を指定してください(3)"
  usage
  exit
fi
# パラメータが配列数未満か確認
if [ ${MODEL_NO} -ge ${num_models} ] ; then
  echo " 0 以上 $num_models 未満の数値を指定してください(2)"
  usage
  exit
fi

# モデル名
MODEL_NAME=${MODEL_NAMES[${MODEL_NO}]}

execute

exit
