# テスト用画像ファイルの保存用ディレクトリ

# ファイルのダウンロード

``download.sh``を実行すると、``ssd``ディレクトリのテスト用スクリプトで使用するデフォルト入力画像をダウンロードする。  
その他、テスト用の画像ファイル(mp4/jpg/png)を適当にダウンロードする(このディレクトリでなくてもいいけど)。  

# メモ

入力画像が大きいと、テスト結果表示時にエライことになるので、適当にリサイズしておくこと。  
(mp4ファイルのリサイズは面倒なので、省略。https://www.videosmaller.com/jp/ あたりが使えるかな？)  

## ツールのインストール  

ubuntu 18.04 では既にインストールされてるかも...  

```bash
sudo apt install imagemagick
```

## 実行

jpegやpngは以下でリサイズできる。  

```bash
convert «入力ファイル» -resize «幅»x«高さ» «出力ファイル»
```

«幅» や «高さ» のどちらかを省略すると アスペクト比を保持したまま、指定した方のサイズに合わせてリサイズしてくれる。  
«幅»x«高さ» ではなく、 % 値 で指定することも可能。  
フォーマット変換も同時にやってくれる(フォーマッは出力ファイルの拡張子で指定)。  

## 例

```bash
convert /work/data/toybear005.jpg -resize 800x teddybear.jpg
convert /work/data/cityscapes2.png -resize 800x sidewalk.jpg
convert /work/data/airplane.jpg -resize 800x airplane.jpg 
```

