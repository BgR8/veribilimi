import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a,b)
with tf.Session() as sess:
    # add this line to use TensorBoard
    writer = tf.summary.FileWriter('C:\tensorflow\graphs', sess.graph)
    print(sess.run)

writer.close()