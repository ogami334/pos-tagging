from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import OrderedDict
import sys
from .model import Model
# sys.path.append("../layers")
from ..common.layers import Affine, SoftmaxWithLoss, Relu, Embedding, SumLine
import logging
logger = logging.getLogger(__name__)

PARAMETER_FILE_NAME = "parameters.npz"
# need to be modified

# やること
# 高速化(流石に遅いので一層目だけ特別扱いする？)
@Model.register("neural_network")
class Neural_Network(Model):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int = 30, weight_init_std: float = 0.01, learning_rate: float = 0.001, num_layers: int = 2):
        super().__init__(num_features, num_classes)
        self.hidden_size = hidden_size
        self.weight_init_std = weight_init_std
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        # 重みパラメータの初期化
        self.params = {}
        self.params['E1'] = weight_init_std * np.random.randn(num_features + 1, hidden_size)
        # integrate bias into weight
        # self.params['b1'] = np.zeros(hidden_size)
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, num_classes)
        self.params['b1'] = np.zeros(num_classes)

        # 各層に何を置くかを決める
        self.layers = OrderedDict()
        self.layers['Embedding'] = Embedding(self.params['E1'])
        self.layers['SumLine'] = SumLine()
        self.layers['Relu1'] = Relu()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.lastLayer = SoftmaxWithLoss()

        # 各層に何を入れるか決めたら自動で初期化して欲しい。。
        # 各層の名前とself.params上でのパラメータの対応づけを考えたい。

    def forward(self, x):
        # xというnp.ndarray(batch_size, num_features)を受け取り、ニューラルネットワークにかけてnp.ndarray(batch_size, num_classes)を返す
        for key, layer in self.layers.items():
            x = layer.forward(x)
        return x

    def transform2d(self, word_features, length, dtype='float64'):
        x = np.zeros(shape=(len(word_features), length), dtype=dtype)
        for index, fs in enumerate(word_features):
            x[index][fs] = 1
        return x

    def predict_batch(self, word_features: List[List[int]]) -> List[int]:
        # for i in range(len(word_features)):
        #     word_features[i].append(self._num_features)
        # print(word_features)
        # word_features = np.array(word_features, dtype='object')
        # y = self.forward(word_features)
        # return y
        # forward関数の結果をまとめて返す
        Y = np.empty((0, self.num_classes), dtype='float64')
        for fs in word_features:
            fs.append(self._num_features)
            fs = np.array([fs])
            y = self.forward(fs)
            y = np.array([y])
            Y = np.append(Y, y, axis=0)
        return Y

    def predict_single(self, fs: List[int]):
        y = self.forward(fs)
        return y

    def loss(self, x, t):
        y = self.predict_single(x)
        # ここまででyがsoftmaxをかける前: np.array(batch_size, class_num)
        #  # tはnp.array(batch_size, class_num)のone-hotベクトルになってて欲しい
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['E1'] = self.layers['Embedding'].dW
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db

        return grads

    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        tags_ohv = self.transform2d(tags, self.num_classes, dtype='int32')
        predicted_labels = []
        for index, fs in enumerate(word_features):
            fs.append(self._num_features)
            fs = np.array([fs])
            y = self.predict_single(fs)
            predicted_labels.append(y.argmax())
            grads = self.gradient(fs, tags_ohv[index])
            for key in grads.keys():
                self.params[key] -= self.learning_rate * grads[key]

        return {"prediction": predicted_labels}

    def save(self, save_directory: str):
        np.savez(Path(save_directory)/ PARAMETER_FILE_NAME, **self.params)

    @classmethod
    def load(cls, save_directory):
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, hidden_size = parameters['E1'].shape
        num_features -= 1
        num_classes = parameters['b1'].shape[1]
        model = Neural_Network(num_features, num_classes, hidden_size, weight_init_std=0.01)
        for key in model.params.keys():
            model.params[key] = parameters[key]
        return model
    # モデルの変更に伴って書き換える必要あり。
    # OrderedDictをうまく使うのが良さそう？
