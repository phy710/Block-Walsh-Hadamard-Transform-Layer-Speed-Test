import numpy as np
import tensorflow as tf
from time import time

def h_sigmoid(x):
    
    output = tf.keras.layers.Activation('hard_sigmoid')(x)
    
    return output

def h_swish(x):
    
    output = x*h_sigmoid(x)
    
    return output

def Squeeze_excitation_layer(x):
    
    inputs = x
    squeeze = inputs.shape[-1]/4
    excitation = inputs.shape[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(squeeze)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(excitation)(x)
    x = h_sigmoid(x)
    x = tf.keras.layers.Reshape((1, 1, excitation))(x)
    x = inputs * x

    return x



num_features = 1024
input_size = 8
num = 10

f_1 = tf.keras.Input(shape=(input_size, input_size, num_features), name="input")
f_2 = Squeeze_excitation_layer(f_1)
model = tf.keras.Model(inputs=f_1, outputs=f_2)
model.compile()
model.summary()

x = np.random.rand(num, input_size, input_size, num_features).astype(np.float32)

start = time()
y = model.predict(x)
end = time()
time1 = end-start

saved_model_path = "./sq/saved_model"
model.save(saved_model_path)

# TODO(b/156102192)
optimize_lite_model = False 

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

lite_model_content = converter.convert()

with open("./sq/lite_model.tflite", "wb") as f:
  f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." %
      ("optimized " if optimize_lite_model else "", len(lite_model_content)))

# interpreter = tf.lite.Interpreter(model_content=lite_model_content)
interpreter = tf.lite.Interpreter(model_path="./sq/lite_model.tflite")

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

