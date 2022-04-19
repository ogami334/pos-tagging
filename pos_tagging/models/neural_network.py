from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import OrderedDict
import sys
from .model import Model
# sys.path.append("../layers")
from ..common.layers import Affine,SoftmaxWithLoss, Relu
import logging
logger = logging.getLogger(__name__)

PARAMETER_FILE_NAME = "parameters.npz"
# need to be modified

# やること
# 高速化(流石に遅いので一層目だけ特別扱いする？)
@Model.register("neural_network")
class Neural_Network(Model):
    def __init__(self, num_features: int, num_classes:int, hidden_size:int=30, weight_init_std:float = 0.01, learning_rate: float =0.001, num_layers:int =2):
        super().__init__(num_features, num_classes)
        self.hidden_size = hidden_size
        self.weight_init_std = weight_init_std
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        # 重みパラメータの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(num_features, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        for i in range(num_layers - 2):
            self.params['W' + str(i+2)] = weight_init_std * np.random.randn(hidden_size, hidden_size)
            self.params['b' + str(i+2)] = weight_init_std * np.random.randn(hidden_size)
        self.params['W' + str(num_layers)] = weight_init_std * np.random.randn(hidden_size, num_classes)
        self.params['b' + str(num_layers)] = np.zeros(num_classes)

        # 各層に何を置くかを決める
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        for i in range(num_layers - 1):
            self.layers['Relu' + str(i + 1)] = Relu()
            self.layers['Affine' + str(i + 2)] = Affine(self.params['W' + str(i+2)], self.params['b' + str(i+2)])
        self.lastLayer = SoftmaxWithLoss()

        # 各層に何を入れるか決めたら自動で初期化して欲しい。。
        # 各層の名前とself.params上でのパラメータの対応づけを考えたい。

    def forward(self, x):
        # xというnp.ndarray(batch_size, num_features)を受け取り、ニューラルネットワークにかけてnp.ndarray(batch_size, num_classes)を返す
        for key, layer in self.layers.items():
            x = layer.forward(x)
        return x

    def transform2d(self, word_features, length, dtype='float64'):
        x = np.zeros(shape=(len(word_features),length), dtype=dtype)
        for index,fs in enumerate(word_features):
            for feature_index in fs:
                x[index][feature_index] = 1
        return x

    def predict(self, word_features: List[List[int]]) -> List[int]:
        # np.arrayを定義する
        # forward関数でバッチ処理をする
        # forward関数からnp.arrayを受け取る
        # それをpredicted_tagsに変換する
        x = self.transform2d(word_features, self._num_features)
        y = self.forward(x)
        predicted_tags = np.argmax(y, axis=1)
        predicted_tags = predicted_tags.tolist()

        return predicted_tags

    def loss(self, x, t):
        y = self.forward(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for i in range(self.num_layers):
            grads['W' + str(i+1)], grads['b' + str(i+1)] = self.layers['Affine' + str(i+1)].dW, self.layers['Affine' + str(i+1)].db
        # grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        # grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        predicted_labels = self.predict(word_features)
        new_tags = []
        for tag in tags:
            new_tags.append([tag])
        x = self.transform2d(word_features, self._num_features)
        new_tags = self.transform2d(new_tags, self.num_classes, dtype='int32')
        grads = self.gradient(x, new_tags)
        for key in grads.keys():
            self.params[key] -= self.learning_rate * grads[key]

        return {"prediction": predicted_labels}

    def save(self, save_directory:str):
        np.savez(Path(save_directory)/ PARAMETER_FILE_NAME, **self.params)
        # np.savez(Path(save_directory)/ PARAMETER_FILE_NAME, W1=self.params['W1'], b1 =self.params['b1'], W2=self.params['W2'], b2=self.params['b2'])

    @classmethod
    def load(cls, save_directory):
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, hidden_size = parameters['W1'].shape
        num_classes = parameters['b2'].shape[1]
        model = Neural_Network(num_features, num_classes, hidden_size, weight_init_std=0.01)
        for key in self.params.keys():
            self.params[key] = parameters[key]
        return model
    # モデルの変更に伴って書き換える必要あり。
    # OrderedDictをうまく使うのが良さそう？
