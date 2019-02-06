import tensorflow as tf
gf = tf.GraphDef()
gf.ParseFromString(open('model_files/frozen_model_my_model.pb', 'rb').read())

#print([n.name + '=>' + n.op for n in gf.node if n.op in ( 'Softmax','Placeholder')])

[print(n.name) for n in gf.node]