# processing
raspi_preprocessing : 元画像をpicとheightに分けてトリミング
raspi_processing : trainとtestに分けてtrainを水増し、np.arrayで保存
raspi_predict : モデル構築、予測

撮影してきた画像を取り込んでからモデルを学習させるまでです。このモデルを5raspiにコピーしてラズパイに入れます。
