import numpy as np
import tensorflow as tf
from tf_data_preprocess import create_feature_sets_and_labels

x_train, y_train, x_test, y_test = create_feature_sets_and_labels(
        pos='./pos.txt',
        neg='./neg.txt',
        test_size=0.1)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

x = tf.placeholder(dtype='float', shape=[None, len(x_train[0])])
y = tf.placeholder(dtype='float')

def neural_network_model(data):
    """
    Takes input sentiment_nn and returns output of the output layer
    :param data:
    :return:
    """
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal(shape=[len(x_train[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal(shape=[n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal(shape=[n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal(shape=[n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal(shape=[n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal(shape=[n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal(shape=[n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal(shape=[n_classes]))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):

            epoch_loss = 0

            i = 0
            while i < len(x_train):
                start = i
                end = i+batch_size
                x_batch = np.array(x_train[start:end])
                y_batch = np.array(y_train[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.arg_max(input=prediction,dimension=1), tf.arg_max(input=y,dimension=1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:x_test, y:y_test}))

train_neural_network(x)




