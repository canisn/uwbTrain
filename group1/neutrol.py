import os
import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases


if __name__ == '__main__':
    # 训练参数
    Gradient = 0.1
    trainStep = 1000
    rawdata_rows = 16000
    rawdata_trainRow = 15000
    batch_size = 100
    rawdata_testRow = rawdata_rows - rawdata_trainRow
    rawdata_trainCol = 23
    rawdata_labelCol = 3
    saver_path = "./"
    CHECHPOINT = trainStep / 10000
    start_time = datetime.datetime.now()
    rawdata = np.loadtxt('../sourceData/Train_Test_Data_8000db_single.csv',
                         dtype=float, delimiter=',',
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 24, 25,
                                  26, 27, 28, 29, 30, 31, 32, 35, 36, 37),
                         skiprows=1, unpack=False, ndmin=0,
                         # converters={39: lambda s: float(s.strip() or 0),
                         #             40: lambda s: float(s.strip() or 0)}
                         )
    rawdata, temp = np.vsplit(rawdata, np.array([rawdata_rows, ]))

    # 归一化方法1
    # temp, rawdata = np.hsplit(rawdata, np.array([7, ]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(rawdata.T)
    rawdata_min = scaler.data_min_
    rawdata_max = scaler.data_max_
    data = scaler.transform(rawdata.T)
    data = data.T
    # 归一化方法2
    # data = preprocessing.scale(rawdata, axis=1)

    # 原始数据
    # data = rawdata

    train, test = np.vsplit(data, np.array([rawdata_trainRow, ]))
    test_data, test_label = np.hsplit(test, np.array([rawdata_trainCol, ]))
    # print('trainDataSize:[%d,%d],[%d,%d]' % (train_data.shape[
    #                                              0], train_data.shape[1], train_label.shape[0], train_label.shape[1]))
    # print('testDataSize:[%d,%d],[%d,%d]' % (test_data.shape[
    #                                             0], test_data.shape[1], test_label.shape[0], test_label.shape[1]))
    print('train_step:%d\ttrain_gradient:%0.4f' % (trainStep, Gradient))
    xs = tf.placeholder(tf.float32, [None, rawdata_trainCol])
    ys = tf.placeholder(tf.float32, [None, rawdata_labelCol])
    l1, w1, b1 = add_layer(xs, rawdata_trainCol, 6, activation_function=tf.nn.sigmoid)
    l2, w2, b2 = add_layer(l1, 6, 6, activation_function=tf.nn.sigmoid)
    l3, w3, b3 = add_layer(l2, 6, 6, activation_function=tf.nn.sigmoid)
    l4, w4, b4 = add_layer(l3, 6, 6, activation_function=tf.nn.sigmoid)
    prediction, w5, b5 = add_layer(l4, 6, rawdata_labelCol, activation_function=None)
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(ys - prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(Gradient).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":8},
                                            inter_op_parallelism_threads=0,
                                            intra_op_parallelism_threads=0))
    sess.run(init)
    saver = tf.train.Saver()
    tf.add_to_collection('prediction', prediction)
    tf.add_to_collection('input_data', xs)
    for i in range(trainStep):
        rand_index = np.random.randint(0, len(train)-batch_size)
        batch = train[rand_index:rand_index + batch_size,:]
        train_data, train_label = np.hsplit(batch, np.array([rawdata_trainCol, ]))
        sess.run(train_step, feed_dict={xs: train_data, ys: train_label})
        if (i + 1) % CHECHPOINT == 0:
            print('current_step:%d\t\ttrain_loss:%.10f\t\ttest_loss:%.10f'
                  % (i + 1, sess.run(loss,
                                     feed_dict={xs: train_data,
                                                ys: train_label}), sess.run(
                loss, feed_dict={xs: test_data, ys: test_label})))

            saver.save(sess, saver_path + 'model')
    np.savetxt(saver_path + 'prediction.csv', sess.run(
        prediction, feed_dict={xs: test_data}), delimiter=',')
    np.savetxt(saver_path + 'loss.csv', sess.run(
        prediction, feed_dict={xs: test_data}) - test_label, delimiter=',')

    # 数据还原
    temp, testLabelMax = np.hsplit(rawdata_max, np.array([rawdata_trainRow, ]))
    temp, testLabelMin = np.hsplit(rawdata_min, np.array([rawdata_trainRow, ]))
    # print(testLabelMax)
    # print(testLabelMin)
    # print(rawdata_max)
    # print(testLabelMax)
    # print(testLabelMin)
    # print(testLabelMax.shape[0])
    final_prediction = np.loadtxt(saver_path + 'prediction.csv',
                                  delimiter=',')
    raw_testlabel = (np.multiply(
        (testLabelMax - testLabelMin), test_label.T) + testLabelMin).T
    prediction_recover = (np.multiply(
        (testLabelMax - testLabelMin), final_prediction.T) + testLabelMin).T
    loss_and_label = np.column_stack(
        (prediction_recover - raw_testlabel, raw_testlabel))
    np.savetxt(saver_path + 'loss_and_label.csv',
               loss_and_label, delimiter=',')
    stop_time = datetime.datetime.now()
    print("startAtTime:%s\nstopAtTime: %s" % (start_time, stop_time))
