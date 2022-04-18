from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import OrderedDict
import sys
from .model import Model
# sys.path.append("../layers")
from ..layers.layers import Affine,SoftmaxWithLoss, Relu
import logging
logger = logging.getLogger(__name__)

PARAMETER_FILE_NAME = "parameters.npz"
# need to be modified

@Model.register("neural_network")
class Neural_Network(Model):
    def __init__(self, num_features: int, num_classes:int, hidden_size:int=30, weight_init_std:float = 0.01):
        super().__init__(num_features, num_classes)
        # 重みパラメータの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(num_features, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        # 各層に何を置くかを決める
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

        #今の感じだと各層の中身と初期パラメータを両方決めなきゃいけないのが辛い。

    def predict(self, word_features: List[List[int]]) -> List[int]:
        #順方向の処理(予測に使うのでsoftmaxがいらない)
        #一層目の処理だけ分けて書くのめんどくさい...
        predicted_tags = []
        for fs in word_features:
            for key, layer in self.layers.items():
                # if key[:3] == "Aff":
                #     # logger.debug(f"layer:{key}, shape:{layer.W.shape}") なぜか表示されない
                #     print(f"layer:{key}, shape:{layer.W.shape}")
                fs = layer.forward(fs)
            predicted_tag = fs.argmax()
            predicted_tags.append(predicted_tag)
        # 逐次処理してるのでまとめて処理するコードを書きたい
        return predicted_tags

    def loss(self, x, t):
        y = self.predict(x)
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
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        predicted_labels = self.predict(word_features)
        # 間違ったものを抜き出す処理をしていない
        grad = self.gradient(word_features, predicted_labels)
        for key in self.layers.keys():
            self.params[key] -= learning_rate * grad[key]

        # 間違った問題について、パラメータを更新する
        return {"prediction": predicted_labels}

    def save(self):
        np.savez(Path(save_directory)/ PARAMETER_FILE_NAME, W1=self.params['W1'], b1 =self.params['b1'], W2=self.params['W2'], b2=self.params['b2'])

    @classmethod
    def load(cls, save_directory):
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, hidden_size = parameters['W1'].shape
        num_classes = parameters['b2'].shape[1]
        model = Neural_Network(num_features, num_classes, hidden_size, weight_init_std=0.01)
        for key in self.layers.keys():
            self.params[key] = parameters[key]
        return model
    # モデルの変更に伴って書き換える必要あり。
    # OrderedDictをうまく使うのが良さそう？
