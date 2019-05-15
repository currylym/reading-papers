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
        # input shape:[(time_step,feature_num1),(feature_num2,)]
        self.Wb = self.add_weight(name='wb', 
                                  shape=(input_shape[0][2], input_shape[1][1]),
                                  initializer=self.initializer,
                                  trainable=self.trainable)
        super(BiLinearAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):

        C,h = inputs # C是上下文，h是当前状态
        if len(K.int_shape(h)) == 3:
            h = K.sum(h,axis=-1)

        # 计算权重
        W = K.dot(C,self.Wb)
        weight = K.batch_dot(W,h,axes=[2,1])
        weight = K.softmax(weight,axis=1)
        
        # 计算输出
        C_reduced = K.batch_dot(C,weight,axes=[1,1])
        print(K.int_shape(C_reduced))

        # 判断维度
        assert K.int_shape(C_reduced)[1] == K.int_shape(C)[-1]
        return C_reduced

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

# 加性注意力
class ConCatAttention(Layer):

    def __init__(self, attention_dim, initializer='uniform',trainable=True, **kwargs):
        self.initializer = initializer
        self.trainable = trainable
        self.attention_dim = attention_dim
        super(ConCatAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # input shape:[(time_step,feature_num1),(feature_num2,)]
        self.Wc1 = self.add_weight(name='wc1', 
                                  shape=(input_shape[0][2], self.attention_dim),
                                  initializer=self.initializer,
                                  trainable=self.trainable)
        self.Wc2 = self.add_weight(name='wc2', 
                                  shape=(input_shape[1][1], self.attention_dim),
                                  initializer=self.initializer,
                                  trainable=self.trainable)
        self.Vc = self.add_weight(name='vc', 
                                  shape=(self.attention_dim,1),
                                  initializer=self.initializer,
                                  trainable=self.trainable)
        super(ConCatAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):

        C,h = inputs # C是上下文，h是当前状态
        time_step = K.int_shape(C)[1]
        if len(K.int_shape(h)) == 3:
            h = K.sum(h,axis=-1)

        # 计算权重
        W1 = K.dot(C,self.Wc1)
        W2 = K.dot(h,self.Wc2)
        W2 = K.repeat(W2,time_step)
        W = W1 + W2
        W = K.tanh(W)
        weight = K.dot(W,self.Vc)
        weight = K.softmax(weight,axis=1)

        # 计算输出
        C_reduced = K.batch_dot(C,weight,axes=[1,1])
        print(K.int_shape(C_reduced))

        # 判断维度
        assert K.int_shape(C_reduced)[1] == K.int_shape(C)[-1]
        return C_reduced

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1], 1)

def SingleHeadSelfAttention(Layer):

    def __init__(self, attention_dim, initializer='uniform',trainable=True, **kwargs):
        self.initializer = initializer
        self.trainable = trainable
        self.attention_dim = attention_dim
        super(ConCatAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        '''
        input shape:
           Q:(n,d_k)
           K:(m,d_k)
           V:(m,d_v)
        '''


if __name__ == '__main__':
    import numpy as np

    def BiLinearAttentionTest():
        # 加性注意力测试
        # 模型参数测试
        Att = BiLinearAttention(initializer='ones')
        X = keras.layers.Input(shape=(2,3))
        h = keras.layers.Input(shape=(4,))
        print(K.int_shape(h))
        Y = Att([X,h])
        model = keras.models.Model(inputs=[X,h],outputs=Y)
        model.summary()
        # 计算结果测试
        X1 = K.variable(np.array([[[1,0,0],[0,1,0]],[[1,0,0],[0,1,0]]]))
        h1 = K.variable(np.array(([[[1],[0],[0],[0]],[[1],[0],[0],[0]]])))
        print(K.int_shape(X1))
        print(K.int_shape(h1))
        y = Att([X1,h1])
        print(K.eval(y))

    BiLinearAttentionTest()

    def ConCatAttentionTest():
        # 乘性注意力测试
        Att1 = ConCatAttention(attention_dim=10)
        X1 = K.variable(np.array([[[1,0,0],[0,1,0]],[[1,0,0],[0,1,0]]]))
        h1 = K.variable(np.array(([[[1],[0],[0],[0]],[[1],[0],[0],[0]]])))
        print(K.int_shape(X1))
        print(K.int_shape(h1))
        y = Att1([X1,h1])
        print(K.eval(y))

    #ConCatAttentionTest()
