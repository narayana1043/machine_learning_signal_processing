import tensorflow as tf
import numpy as np
import pickle
from nltk import word_tokenize
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

##########################
n_classes = 2
# 200 data? 52% (10 epcohs)
# 2000 data? 62% (10 epcohs) 9-10 sec on GPU tf: -14 sec on CPU
# 2000 data? 53% (15 epcohs)
hm_data = 2000000
##########################

batch_size = 32
total_batches =  int(1600000/batch_size)
hm_epochs = 1

x = tf.placeholder(dtype='float')
y = tf.placeholder(dtype='float')

current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weights': tf.Variable(tf.random_normal(shape=[2638, n_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal(shape=[n_nodes_hl1]))}
hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weights': tf.Variable(tf.random_normal(shape=[n_nodes_hl2, n_nodes_hl2])),
                  'biases': tf.Variable(tf.random_normal(shape=[n_nodes_hl2]))}
output_layer = {'weights': tf.Variable(tf.random_normal(shape=[n_nodes_hl2, n_classes])),
                'biases': tf.Variable(tf.random_normal(shape=[n_classes]))}

# Important
saver = tf.train.Saver()
tf_log = 'tf.log'

def neural_network_model(data):
    """
    Takes input sentiment_nn and returns output of the output layer
    :param data:
    :return:
    """

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
            print('STARTING:', epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:

            if epoch != 1:
                saver.restore(sess, "model.ckpt")
            epoch_loss = 1

            with open('lexicon-2500-2638.pickle', 'rb') as f:
                lexicon = pickle.load(f)

            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                x_batch = []
                y_batch = []
                batches_run = 0

                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1

                    line_x = list(features)
                    line_y = eval(label)

                    x_batch.append(line_x)
                    y_batch.append(line_y)

                    if len(x_batch) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(x_batch),
                                                                      y: np.array(y_batch)})
                        epoch_loss += c
                        x_batch = []
                        y_batch = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:',
                              epoch, '| Batch Loss:', c, )

            saver.save(sess, "model.ckpt")
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            with open(tf_log, 'a') as f:
                f.write(str(epoch)+'\n')
            epoch += 1

        correct = tf.equal(tf.arg_max(input=prediction, dimension=1), tf.arg_max(input=y, dimension=1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Train Accuracy:', accuracy)
        # feature_sets = []
        # labels = []
        # counter = 0
        #
        # with open('processed=test-set-2500-2638.csv', buffering=20000) as f:
        #     for line in f:
        #         try:
        #             features = list(eval(line.split('::')[0]))
        #             label = list(eval(line.split('::')[1]))
        #
        #             feature_sets.append(features)
        #             labels.append(label)
        #             counter += 1
        #         except:
        #             pass
        #
        #     print("Tested", counter, 'samples.')
        #
        #     x_test = np.array(feature_sets)
        #     y_test = np.array(labels)
        #
        # print('Accuracy:', accuracy.eval({x:x_test, y:y_test}))


def test_neural_network():

    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            try:
                saver.restore(sess, "model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy  = tf.reduce_mean((tf.cast(correct, 'float')))
        feature_sets = []
        labels = []
        counter = 0

        with open('processed-test-set-2500-2638.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))

                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass

            print("Tested", counter, 'samples.')

            x_test = np.array(feature_sets)
            y_test = np.array(labels)

        print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))


def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
        print((prediction.eval(feed_dict={x: [features]}), 1))

        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)


train_neural_network(x)
test_neural_network()

use_neural_network("He's an idiot and a jerk.")
use_neural_network("This was the best store i've ever seen.")

