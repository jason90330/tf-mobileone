import os
from mobileone import *
import tensorflow as tf

model = mobileone(variant='s0')
model.build(input_shape=(None, 224, 224, 3))
print(model)

i = tf.random.normal((4, 224, 224, 3))
o = model(i)
# print(o)

# reparam_model = reparameterize_model(model)
# print(reparam_model)


deploy_model = reparameterize_model(model=model, variant='s0',  input_size=(224,224,3), save_path="reparam")

deploy_converter = tf.lite.TFLiteConverter.from_keras_model(deploy_model)
tflite_deploy_model = deploy_converter.convert()

model_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = model_converter.convert()

# Save the TFLite model to a file
os.makedirs("./tflite", exist_ok=True)
with open("./tflite/reparam_model.tflite", "wb") as f:
    f.write(tflite_deploy_model)
with open("./tflite/model.tflite", "wb") as f:
    f.write(tflite_model)