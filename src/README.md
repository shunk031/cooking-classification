# Script Detail

## `preprocess/crop.py`

`dataset/labeled`以下にある画像データに対して、256 x 256サイズにクロッピングし、 `dataset/cropped_images/train_model`に保存する。

* 引数オプション
  * --cae  
	CAEの学習/テスト用に`dataset/unlabeled`以下にある画像データに対して処理を行い、 `dataset/cropped_images/train_cae`に保存する。

## `preprocess/dump_image_list.py`

Modelの学習/テスト用に`[(image filename, class labele), ...]`のような形式のリストを生成し、 `labeled_image_dataset_list.pkl`として保存する。

* 引数オプション
  * --cae  
	CAEの学習/テスト用に`[(image filename 1, 0), (image filename 2, 0), ...]`のような形式のリストを生成し、 `cae_image_dataset_list.pkl`として保存する。

## `preprocess/compute_mean.py`

`dataset/cropped_images/train_model`からクロップされた256 x 256サイズの画像を読み込んで平均画像を生成し、 `train_mean.npy`として保存する。

* 引数オプション
  * --root  
	読み込む画像ファイルのルートディレクトリを指定する。
  * --cae  
	CAEの学習用に`dataset/cropped_images/train_cae`から画像を読み込んで処理を行い、`cae_mean.npy`として保存する。
  * --image
	生成した平均画像をJPEG画像として出力する。

``` shell
# Modelを学習させるための平均画像を生成し、画像としても出力する。
$ python compute_mean.py labeled_image_dataset_list.pkl --root /path/to/cropped_images/train_model/ --image
# CAEを学習させるための平均画像を生成する。
$ python compute_mean.py cae_image_dataset_list.pkl --root /path/to/cropped_images/train_cae/ --cae
```

## `train_cae/autoencoder.py`

Auto-encoderモデルを定義している。

## `train_cae/train_autoencoder`

Auto-encoderの学習を行うスクリプトである。

``` shell
$ python train_autoencoder.py cae_image_dataset_list.pkl --batchsize 32 --epoch 100 --gpu 1 --loaderjob 4 --mean cae_mean.npy --root /path/to/cropped_images/train_cae --val_batchsize 250
```

## `model.py`

識別モデルを定義している。

### `train.py`

識別モデルの学習を行うスクリプトである。

``` shell
$ python train.py labeled_image_dataset_list.pkl --arch deepalexlike --batchsize 32 --epoch 100 --gpu 1 --initmodel cae_model_final.npz --loaderjob 4 --mean train_mean.npy --root /path/to/cropped_images/train_model/ --val_batchsize 250 
```
