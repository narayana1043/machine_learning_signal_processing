import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

print(result)


# right way to use tensor flow
sess = tf.Session()
print(sess.run(result))

# in case of vectors or matrices
x1 = tf.constant([[5]])
x2 = tf.constant([[6]])

result = tf.matmul(x1, x2)
print(sess.run(result))
