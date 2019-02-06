import tensorflow as tf

grap_def_file = "model_files/frozen_my_model.pb" # the .pb file

input_arrays = ["X"] #Input node
output_arrays = ["Y"] #Output node

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    grap_def_file, input_arrays, output_arrays
)

tflite_model = converter.convert()

open("MobileNet/my_model.tflite", "wb").write(tflite_model)