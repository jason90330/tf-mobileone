from mobileone import *
import tensorflow as tf

model = mobileone(variant='s0')
model.build(input_shape=(None, 224, 224, 3))
print(model)

i = tf.random.normal((4, 224, 224, 3))
o = model(i)
print(o)

# reparam_model = reparameterize_model(model)
# print(reparam_model)


deploy_model = reparameterize_model(model=model, variant='s0',  input_size=(224,224,3), save_path="reparam")