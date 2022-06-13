import numpy as np
import tensorflow as tf
from time import time

def hadamard_transform(u, axis=-1):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = tf.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = tf.concat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), axis=-1)
    y = tf.squeeze(x, axis=-2) / 2**(m / 2)
    
    if axis != -1:
        y = tf.transpose(y, -1, axis)
        
    return y

class SoftThreshold(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super(SoftThreshold, self).__init__(**kwargs)
        self.num_features = num_features

    def build(self, input_shape):
        #self.variable = tf.Variable(np.random.uniform(0., 0.1, self.num_features), trainable=True, dtype=tf.float32)
        self.variable = tf.Variable(np.arange(0, self.num_features)/(10*self.num_features), trainable=True, dtype=tf.float32)
        self.w = tf.Variable(np.ones(self.num_features), trainable=True, dtype=tf.float32)

    def get_config(self):
        config = super(SoftThreshold, self).get_config()
        config.update({"num_features": self.num_features})
        return config

    def call(self, inputs):
        x = tf.math.multiply(inputs, self.w)
        a = tf.math.abs(x)
        r = tf.keras.layers.ReLU()(a-self.variable)
        t = tf.math.tanh(x)
        return tf.keras.layers.Multiply()([t, r])
    
class SoftThreshold2(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super(SoftThreshold2, self).__init__(**kwargs)
        self.num_features = num_features

    def build(self, input_shape):
        #self.variable = tf.Variable(np.random.uniform(0., 0.1, self.num_features), trainable=True, dtype=tf.float32)
        v = np.zeros((self.num_features, self.num_features))
        for i in range(self.num_features):
            v[:, i] = (i + np.arange(self.num_features))/(10*self.num_features);
        self.variable = tf.Variable(v, trainable=True, dtype=tf.float32)
        self.w = tf.Variable(np.ones((self.num_features,self.num_features)), trainable=True, dtype=tf.float32)

    def get_config(self):
        config = super(SoftThreshold2, self).get_config()
        config.update({"num_features": self.num_features})
        return config

    def call(self, inputs):
        x = tf.math.multiply(inputs, self.w)
        a = tf.math.abs(x)
        r = tf.keras.layers.ReLU()(a-self.variable)
        t = tf.math.tanh(x)
        return tf.keras.layers.Multiply()([t, r])
    
    
def fwht_layer(x, stride = 1, num_features = 0, out_num_features = 0, Hadamard_size = 0):
    x = x[:, ::stride, ::stride, :]
    if num_features<=0:
        num_features = x.shape.as_list()[-1]
    if out_num_features<=0:
        out_num_features = x.shape.as_list()[-1]  
    if Hadamard_size < 1 or Hadamard_size>num_features:
        Hadamard_size = num_features
    if num_features < out_num_features:
        c_0 = x
        s = c_0.shape.as_list()[1]
        c_1 = tf.expand_dims(c_0, -2)
        lis = []
        for i in np.linspace(0, num_features-Hadamard_size, num=out_num_features//Hadamard_size).astype(int):
            lis.append(c_1[:, :, :, :, i:i+Hadamard_size])
        c_2 = tf.concat(lis, axis=-2)
        c_3 = hadamard_transform(c_2)
        c_4 = SoftThreshold(Hadamard_size)(c_3)
        c_5 = hadamard_transform(c_4)
        c_6 = tf.keras.layers.Reshape((s, s, out_num_features))(c_5)
        y = c_6
    else:
        channel_change = num_features//out_num_features
        c_0 = x
        s = c_0.shape.as_list()[1]
        c_1 = tf.keras.layers.Reshape((s, s, num_features//Hadamard_size, Hadamard_size))(c_0)
        c_2 = hadamard_transform(c_1)
        c_3 = SoftThreshold(Hadamard_size)(c_2)
        c_4 = hadamard_transform(c_3)
        c_5 = tf.keras.layers.Reshape((s, s, num_features))(c_4)
        if channel_change == 1:
            c_6 = c_5
        else:
            c_6 = tf.keras.layers.AveragePooling2D(pool_size=(1, channel_change), strides=(1, channel_change), padding='valid', data_format='channels_first')(c_5)
        y = c_6[:, :, :, :out_num_features]
    return y

def find_min_power(x, p=2):
    y = 1
    while y<x:
        y *= p
    return y

def fwht2_layer(x, stride = 1, channel_change = 1, Hadamard_size = 0):
    if channel_change != 1:
        x = fwht_layer(x, channel_change=channel_change, Hadamard_size=Hadamard_size)
    input_size = x.shape.as_list()[-2]
    if Hadamard_size < 1:
        Hadamard_size = input_size
    Hadamard_size = find_min_power(Hadamard_size, 2)
    if Hadamard_size>input_size:
        f_1 = tf.pad(x, [[0, 0], [0, Hadamard_size-input_size], [0, Hadamard_size-input_size], [0, 0]])
    else:
        f_1 = x
    f_2 = tf.transpose(f_1, [0, 3, 2, 1]) #0123->0331
    
    f_3 = hadamard_transform(f_2)
    # f_3 = tf.tensordot(f_2, w, [[-1], [0]])
    f_4 = tf.transpose(f_3, [0, 1, 3, 2]) #0321->0312
    f_5 = hadamard_transform(f_4)
    # f_5 = tf.tensordot(f_4, w, [[-1], [0]])
    f_6 = SoftThreshold2(Hadamard_size)(f_5)
    
    f_7 = hadamard_transform(f_6)
    # f_7 = tf.tensordot(f_6, w, [[-1], [0]])
    f_8 = tf.transpose(f_7, [0, 1, 3, 2]) #0312->0321
    
    
    f_9 = hadamard_transform(f_8)
    # f_9 = tf.tensordot(f_8, w, [[-1], [0]])
    y = tf.transpose(f_9, [0, 3, 2, 1]) #0321->0123
    if stride > 1:
        y = tf.keras.layers.AveragePooling2D(pool_size=(stride, stride))(y)
    if Hadamard_size>input_size:
        y = y[:, :input_size, :input_size, :]
    return y

num_features = 1024
input_size = 8
num = 10

f_1 = tf.keras.Input(shape=(input_size, input_size, num_features), name="input")
f_2 = fwht2_layer(f_1)
model = tf.keras.Model(inputs=f_1, outputs=f_2)
model.compile()
model.summary()

x = np.random.rand(num, input_size, input_size, num_features).astype(np.float32)

start = time()
y = model.predict(x)
end = time()
time1 = end-start

saved_model_path = "./fwht/saved_model"
model.save(saved_model_path)

# TODO(b/156102192)
optimize_lite_model = False 

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

lite_model_content = converter.convert()

with open("./fwht/lite_model.tflite", "wb") as f:
  f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." %
      ("optimized " if optimize_lite_model else "", len(lite_model_content)))

# interpreter = tf.lite.Interpreter(model_content=lite_model_content)
interpreter = tf.lite.Interpreter(model_path="./fwht/lite_model.tflite")

yy = np.zeros((num, input_size, input_size, num_features)).astype(np.float32)
start = time()
for i in range(num):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x[i:i+1, :, :])
    interpreter.invoke()
    yy[i:i+1, :, :] = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end = time()
time2 = end-start

print(time1)
print(time2)