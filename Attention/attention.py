'''
keras version
'''
import keras
from keras import backend as K
from keras.engine.topology import Layer

# 乘法注意力
class BiLinearAttention(Layer):

    def __init__(self, initializer='uniform',trainable=True, **kwargs):
        self.initializer = initializer
        self.trainable = trainable
        super(BiLinearAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # input shape:[(time_step,feature_num1),(1,feature_num2)]
        self.Wb = self.add_weight(name='wb', 
                                  shape=(input_shape[0][1], input_shape[1][1]),
                                  initializer=self.initializer,
                                  trainable=self.trainable)
        super(BiLinearAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        C,h = inputs # C是上下文，h是当前状态
        # 计算权重
        W = K.dot(C,self.Wb)
        h1 = K.reshape(h,shape=(-1,1))
        weight = K.batch_dot(W,h1,axis=[2,1])
        weight = K.softmax(weight)
        print(K.int_shape(weight))
        print(K.eval(weight))
        # 计算输出
        C_reduced = K.batch_dot(self.Wb,weight,axis=[1,1])
        C_reduced = K.sum(C_reduced,axis=1)
        assert K.int_shape(C_reduced)[-1] == K.int_shape(C)[-1]
        return C_reduced

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

if __name__ == '__main__':
    Att = BiLinearAttention()
    X = keras.layers.Input(shape=(20,30))
    h = keras.layers.Input(shape=(30,))
    Y = Att([X,h])
    model = keras.models.Model(X,Y)
    model.summary()
