from distutils.errors import PreprocessError
from pathlib import Path
from typing import Dict, List

from pyparsing import makeXMLTags
import numpy as np
from collections import OrderedDict
import sys
from .model import Model
# sys.path.append("../layers")
from ..common.layers import Affine, SoftmaxWithLoss, Relu, Embedding, SumLine
from ..common.optimizer import SGD, Adam
import logging
logger = logging.getLogger(__name__)

PARAMETER_FILE_NAME = "parameters.npz"
# need to be modified

#型ヒントをかこう
@Model.register("neural_network")
class Neural_Network(Model):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int = 30, weight_init_std: float = 0.01, learning_rate: float = 0.001, num_layers: int = 2):
        super().__init__(num_features, num_classes)
        self.hidden_size = hidden_size
        self.weight_init_std = weight_init_std
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.optimizer = SGD(lr=learning_rate)
        # self.optimizer = Adam(lr=learning_rate)
        # 重みパラメータの初期化
        self.params = {}

        # Heの初期値
        self.params['E1'] = np.sqrt(2 / num_features) * np.random.randn(num_features + 2, hidden_size)
        self.params['E1'][-2] = np.zeros(shape=(1, hidden_size), dtype='float') #　バイアスを0にするのと等価
        self.params['E1'][-1] = np.zeros(shape=(1, hidden_size), dtype='float') # 可変長を固定長にするための行 forwardがよばれる前に0にする

        # Heの初期値
        self.params['W1'] = np.sqrt(2 / hidden_size) * np.random.randn(hidden_size, num_classes)
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
        # print(x.shape)
        for key, layer in self.layers.items():
            x = layer.forward(x)
            # print(x.shape)
        # print("forward finished")
        return x

    def transform2d(self, word_features, length, dtype='float64'):
        x = np.zeros(shape=(len(word_features), length), dtype=dtype)
        for index, fs in enumerate(word_features):
            x[index][fs] = 1
        return x

    def infer(self, processed_word_features: List[List[int]]) -> List[int]:
        Y = self.forward(processed_word_features)
        return Y

    def predict(self, word_features: List[List[int]]) -> List[int]:
        processed_word_features = self.preprocess(word_features)
        infer_result = self.infer(processed_word_features)
        predicted_labels = infer_result.argmax(axis=1).tolist()
        return predicted_labels

    def loss(self, x, t):
        y = self.infer(x)
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
        # print(dout.shape)
        for layer in layers:
            dout = layer.backward(dout)
            # print(dout.shape)
        # print("backward finished")
        grads = {}
        grads['E1'] = self.layers['Embedding'].dW
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        return grads

    def preprocess(self, word_features):
        # 入力長を揃えるための前処理
        max_features = -1
        for i in range(len(word_features)):
            word_features[i].append(self._num_features)
            max_features = max(max_features, len(word_features[i]))
        for i in range(len(word_features)):
            for j in range(max_features - len(word_features[i])):
                word_features[i].append(self._num_features + 1)
        word_features = np.array(word_features)
        return word_features

    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        tags_ohv = self.transform2d(tags, self.num_classes, dtype='int32')
        processed_word_features = self.preprocess(word_features)
        predicted_labels = self.predict(word_features)

        grads = self.gradient(processed_word_features, tags_ohv)
        self.optimizer.update(params=self.params, grads=grads)
        # for key in grads.keys():
        #     self.params[key] -= self.learning_rate * grads[key]
        self.params['E1'][-1] = np.zeros(shape=(1, self.hidden_size), dtype='float')

        return {"prediction": predicted_labels}

    def save(self, save_directory: str):
        np.savez(Path(save_directory) / PARAMETER_FILE_NAME, **self.params)

    @classmethod
    def load(cls, save_directory):
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, hidden_size = parameters['E1'].shape
        num_features -= 2
        num_classes = parameters['b1'].shape[0]
        model = Neural_Network(num_features, num_classes, hidden_size, weight_init_std=0.01)
        for key in model.params.keys():
            assert model.params[key].shape == parameters[key].shape
            model.params[key] = parameters[key]
        return model
    # モデルの変更に伴って書き換える必要あり。
    # OrderedDictをうまく使うのが良さそう？
