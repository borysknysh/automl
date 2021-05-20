import tensorflow as tf

model = tf.keras.models.load_model('saved_model')

model.summary()

print(model.inputs, model.outputs)

for el in model.layers:
    print(el)
