# Train Network

## `crop.py`

`dataset/clf_train_images_labeled_1/`と`dataset/clf_train_images_labeled_2/`にある学習データに対して、256x256サイズにクロッピングし、`/dataset/cropped_images/`に保存する。

## `compute_mean.py`

`dataset/cropped_images/`から平均画像を生成し、mean.npyとして保存する。-iオプションをつけると、生成した平均画像を画像として出力する。

## `model.py`

今回利用するモデルを定義している。

## `train.py`

学習を行う際に実行する。
