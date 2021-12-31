import numpy as np
import tensorflow as tf
from time import time

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
    
def fwht(x):
    f_1 = tf.concat([x[:, :4, :, :]+x[:, 4:, :, :], x[:, :4, :, :]+x[:, 4:, :, :]], axis = 1)
    f_2 = tf.concat([f_1[:, :2, :, :]+f_1[:, 2:4, :, :], f_1[:, :2, :, :]-f_1[:, 2:4, :, :], f_1[:, 4:6, :, :]+f_1[:, 6:, :, :], f_1[:, 4:6, :, :]-f_1[:, 6:, :, :]], axis = 1)
    f_3 = tf.concat([f_2[:, :1, :, :]+f_2[:, 1:2, :, :], f_2[:, :1, :, :]-f_2[:, 1:2, :, :], f_2[:, 2:3, :, :]+f_2[:, 3:4, :, :], f_2[:, 2:3, :, :]-f_2[:, 3:4, :, :], f_2[:, 4:5, :, :]+f_2[:, 5:6, :, :], f_2[:, 4:5, :, :]-f_2[:, 5:6, :, :], f_2[:, 6:7, :, :]+f_2[:, 7:, :, :], f_2[:, 6:7, :, :]-f_2[:, 7:, :, :]], axis = 1)
    y = f_3/np.sqrt(8)
    return y

def fwht2(x):
    f_1 = fwht(x)
    f_2 = tf.transpose(f_1, [0, 2, 1, 3]) #0123->0213
    f_3 = fwht(f_2)   
    f_4 = tf.transpose(f_3, [0, 2, 1, 3]) #0123->0213
    return f_4

def fwht2_layer(x):
    f_1 = fwht(x)
    f_2 = tf.transpose(f_1, [0, 2, 1, 3]) #0123->0213
    f_3 = fwht(f_2)   
    f_4 = tf.transpose(f_3, [0, 3, 2, 1]) #0213->0312
    f_5 = SoftThreshold2(8)(f_4)
    f_6 = tf.transpose(f_5, [0, 3, 2, 1]) #0312->0213
    f_7 = fwht(f_6) 
    f_8 = tf.transpose(f_7, [0, 2, 1, 3]) #0213->0123
    f_9 = fwht(f_8)
    return f_9

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

saved_model_path = "./fwht_add/saved_model"
model.save(saved_model_path)

# TODO(b/156102192)
optimize_lite_model = False 

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

lite_model_content = converter.convert()

with open("./fwht_add/lite_model.tflite", "wb") as f:
  f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." %
      ("optimized " if optimize_lite_model else "", len(lite_model_content)))

# interpreter = tf.lite.Interpreter(model_content=lite_model_content)
interpreter = tf.lite.Interpreter(model_path="./fwht_add/lite_model.tflite")

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