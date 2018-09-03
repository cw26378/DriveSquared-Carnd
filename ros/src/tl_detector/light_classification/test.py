import tensorflow as tf

graph_ob = tf.Graph()

with tf.Session(graph=graph_ob) as sess:
	od_graph_def = tf.GraphDef()
	od_graph_def.ParseFromString(tf.gfile.GFile('model_sim.pb', 'rb').read())
	tf.import_graph_def(od_graph_def, name='')

