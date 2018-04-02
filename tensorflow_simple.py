"""
Minimize the cost function (w-5)^2
"""
import numpy as np
import tensorflow as tf


w = tf.Variable(0, dtype=tf.float32)
cost = tf.add(tf.add(w**2, tf.multiply(-10.0, w)), 25)

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    print(session.run(w))

    session.run(train)
    print(session.run(w))

    for i in range(1000):
        session.run(train)

    print(session.run(w))
