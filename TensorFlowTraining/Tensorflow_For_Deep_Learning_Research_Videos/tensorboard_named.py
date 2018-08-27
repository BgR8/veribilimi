import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a,b, name="add")
with tf.Session() as sess:
    # add this line to use TensorBoard
    writer = tf.summary.FileWriter('C:\\Users\\toshiba\\SkyDrive\\veribilimi.co\\codes\\TensorFlowTraining\\Tensorflow_For_Deep_Learning_Research_Videos\\graphs', sess.graph)
    print(sess.run)

writer.close()