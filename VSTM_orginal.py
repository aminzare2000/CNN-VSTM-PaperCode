import h5py
import numpy as np
import tensorflow as tf


Training = False
ContinueTrain = False
StartFromEpoch = 0
epochs = 10
save_result_path = "E:\\TF_Model_KTH\\saveModel\\"

# expDir = 'd:\\dataset_phist\\kth_phist\\subtract_phist_newTHR_timewindowshift_tensor\\'
expDir = 'D:\\dataset_phist\\kth_phist\\MVSTM_subtract_phist_newTHR_timewindowshift\\'

fixedheight = 136
windowsize = 68
step = 5
nAug_per_file = np.floor((fixedheight - windowsize) / (step - 1)) + 1
fname = 'subtract_phist_newTHR'
k800 = 500

img_h = windowsize
img_w = 490  # feature length of KTH dataset

# img_w = windowsize
# img_h = 490  # feature length of KTH dataset

img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
n_classes = 6  # Number of classes, one class per digit
n_channels = 1
keep_prob_value = 0.5  # keep dropout probility

log_path = "log"
lr = 0.001
batch_size = 32
test_batch_size = 180
display_freq = 50

# 1st Convolutional Layer
filter_size1 = 3  # Convolution filters are 5 x 5 pixels.
num_filters1 = 32  # There are 16 of these filters.
stride1 = 1  # The stride of the sliding window
ksizepooling1 = 3
stridepooling1 = 2

# 2nd Convolutional Layer
filter_size2 = 3  # Convolution filters are 5 x 5 pixels.
num_filters2 = 64  # There are 32 of these filters.
stride2 = 1  # The stride of the sliding window
ksizepooling2 = 3
stridepooling2 = 2


# 3rd Convolutional Layer
filter_size3 = 3
num_filters3 = 128
stride3 = 1  # The stride of the sliding window
ksizepooling3 = 3
stridepooling3 = 2


# 4th Convolutional Layer
filter_size4 = 3
num_filters4 = 256
stride4 = 1  # The stride of the sliding window
ksizepooling4 = 3
stridepooling4 = 2


# 5th Convolutional Layer
filter_size5 = 3
num_filters5 = 1024
stride5 = 1  # The stride of the sliding window
ksizepooling5 = 3
stridepooling5 = 2

# Fully-connected layer.
h1 = 1024  # Number of neurons in fully-connected layer.

def reformat(x, lbl, num_class):
    """
    Reformats the data to the format acceptable for convolutional layers
    :param x: input array
    :param lbl: corresponding labels
    :return: reshaped input and labels
    """
    num_ch = 1
    dataset = x.T.reshape((-1, img_w, img_h, num_ch)).astype(np.float32)\
        .transpose(0, 2, 1, 3)
    labels = (np.arange(1, num_class+1, 1) == lbl[:, None]).astype(np.float32)
    return dataset, labels

def randomize(x, lbl):
    permutation = np.random.permutation(lbl.shape[0])
    x_shuffle = x[permutation, :, :, :]
    lbl_shuffle = lbl[permutation]
    return x_shuffle, lbl_shuffle

def get_next_batch(x, lbl, start, end):
    if end >= x.shape[0]:
        end = x.shape[0]
    x_batch = x[start:end, :, :, :]
    lbl_batch = lbl[start:end]
    return x_batch, lbl_batch


def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    # initer = tf.truncated_normal(shape=shape)
    return tf.get_variable('W', shape=shape, dtype=tf.float32, initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b', dtype=tf.float32, initializer=initial)


def conv_layer(x, filter_size, num_filters, stride, phase_train, name):
    """
    Create a 2D convolution layer
    :param x: input from previous layer
    :param filter_size: size of each filter
    :param num_filters: number of filters (or output feature maps)
    :param stride: filter stride
    :param name: layer name
    :return: The output array
    """
    with tf.variable_scope(name):
        num_in_channel = x.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        W = weight_variable(shape)
        tf.summary.histogram('weight', W)
        b = bias_variable(shape=[num_filters])
        tf.summary.histogram('bias', b)
        layer = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
        # layer = tf.nn.depthwise_conv2d()   # https://arxiv.org/abs/1512.00567
        # layer = tf.nn.separable_conv2d()
        layer += b
        bn_layer = tf.layers.batch_normalization(inputs=layer, training=phase_train)
        relu_layer = tf.nn.relu(bn_layer)
        # ret_layer = tf.nn.lrn(relu_layer, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
        return relu_layer


def max_pool(x, ksize, stride, name):
    """
    Create a max pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param stride: stride of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="VALID",
                          name=name)


def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(shape=[in_dim, num_units])
        tf.summary.histogram('weight', W)
        b = bias_variable(shape=[num_units])
        tf.summary.histogram('bias', b)
        drop_x = tf.cond(phase_train, lambda: tf.nn.dropout(x, keep_prob=keep_prob),
                         lambda: x)  # tf.cond or by keep_prob=1
        layer = tf.matmul(drop_x, W)
        # drop_x = tf.nn.dropout(x, keep_prob=keep_prob)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer


with tf.name_scope('Input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, img_h, img_w, n_channels], name='X')
    lbl = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='LBL')
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

# Create the network layers
conv1 = conv_layer(x, filter_size=filter_size1, num_filters=num_filters1, stride=stride1, phase_train=phase_train,
                   name='conv1')
pool1 = max_pool(conv1, ksize=ksizepooling1, stride=stridepooling1, name='pool1')

conv2 = conv_layer(pool1, filter_size=filter_size2, num_filters=num_filters2, stride=stride2, phase_train=phase_train,
                   name='conv2')
pool2 = max_pool(conv2, ksize=ksizepooling2, stride=stridepooling2, name='pool2')

conv3 = conv_layer(pool2, filter_size=filter_size3, num_filters=num_filters3, stride=stride3, phase_train=phase_train,
                   name='conv3')
pool3 = max_pool(conv3, ksize=ksizepooling3, stride=stridepooling3, name='pool3')

conv4 = conv_layer(pool3, filter_size=filter_size4, num_filters=num_filters4, stride=stride4, phase_train=phase_train,
                   name='conv4')
pool4 = max_pool(conv4, ksize=ksizepooling4, stride=stridepooling4, name='pool4')

conv5 = conv_layer(pool4, filter_size=filter_size5, num_filters=num_filters5, stride=stride5, phase_train=phase_train,
                   name='conv5')

layer_flat = flatten_layer(conv5)
fc1 = fc_layer(layer_flat, num_units=h1, name='FC1', use_relu=True)
output_logits = tf.nn.softmax(fc_layer(fc1, n_classes, name='OUT', use_relu=False))
# Error output_logits = fc_layer(fc1, n_classes, name='OUT', use_relu=False)

# Define the loss function, optimizer, accuracy, and predicted class
with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=lbl, logits=output_logits), name='loss')
        tf.summary.scalar('loss', loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # is critical for batch normalization
    with tf.control_dependencies(update_ops):                # is critical for batch normalization
        with tf.variable_scope('Optimizer'):
            # Optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-optim').minimize(loss)
            Optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, name='Adam-optim').minimize(loss)
    with tf.variable_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(lbl, 1), name='correct_predict')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    with tf.variable_scope('Prediction'):
        cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

cl_path = expDir+'cl_resize'+str(fixedheight)+'_window'+str(windowsize)+'_step' + \
          str(step-1)+'_normal_'+fname+'.mat'
cl_mat = h5py.File(cl_path)
print(cl_mat['cl'].dtype)
print(cl_mat['cl'].shape)

co, r1 = cl_mat['cl'].shape
np_cl_mat = np.array(cl_mat['cl'])
new_fid = co-4-1  # python is zero based index
file_id = co-3-1
conditions_id = co-2-1
person_id = co-1-1
class_id = co-1

file_num = len(np.unique(np_cl_mat[new_fid, :]))
record_num = len(np.unique(np_cl_mat[file_id, :]))
clnum = len(np.unique(np_cl_mat[class_id, :]))
# Load KTH database files
train_person = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 1, 4]
test_person = [22, 2, 3, 5, 6, 7, 8, 9, 10]
test_ind = np.zeros(file_num, np.bool)
for k in range(len(test_person)):
    tp = test_person[k]
    tem_test_ind = np_cl_mat[person_id, :] == tp
    test_ind = test_ind | tem_test_ind
train_ind = ~test_ind
sum1 = np.sum(test_ind == True)
# Number of training iterations in each epoch
global_step = 0
Batch_Number = np.int(np.floor(record_num) / k800)  # 5
cntBatch = Batch_Number + 1

saver = tf.train.Saver(max_to_keep=1000)
################################TRAINING###################################################
if Training == True:
    sess = tf.Session()
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    sess.run(init)
    if ContinueTrain == True:
        print("Model " + save_result_path + "model_dp{0:.2f}_ep{1:d}.ckpt".format(1 - keep_prob_value,
                                                                                  StartFromEpoch - 1) + " is loaded")
        saver.restore(sess,
                      save_result_path + "model_dp{0:.2f}_ep{1:d}.ckpt".format(1 - keep_prob_value, StartFromEpoch - 1))
    summery_writer = tf.summary.FileWriter(logdir=log_path, graph=sess.graph)
    for epoch in range(StartFromEpoch, epochs):
        sum_acc_train = 0
        counter_batch = 0
        print('Training epoch: {}'.format(epoch + 1))
        for i in range(1, cntBatch+1):
            train_cl_ind = np.zeros(file_num, np.bool)
            data_path = expDir+'data_resize'+str(fixedheight)+'_window'+str(windowsize)+'_step'+str(step-1)+'_normal_'+fname + \
                        '_' + str(i)+'.mat'
            data_mat = h5py.File(data_path)
            c22, rd = data_mat['data'].shape
            print('DataFile '+str(i)+' size = '+str(rd))
            ss = int(k800 * nAug_per_file)
            if i <= cntBatch:
                batch_train_ind = train_ind[(i - 1) * ss: i * ss]
                if np.sum(batch_train_ind == True) == 0:
                    continue
                train_cl_ind[(i - 1) * ss: i * ss] = batch_train_ind
                trainlbl = np_cl_mat[class_id, train_cl_ind]
            else:
                batch_train_ind = train_ind[(i - 1) * ss:]
                if np.sum(batch_train_ind == True) == 0:
                    continue
                train_cl_ind[(i - 1) * ss:] = batch_train_ind
                trainlbl = np_cl_mat[class_id, train_cl_ind]

            r = np.sum(train_cl_ind == True)
            np_data_mat = np.array(data_mat['data'])
            train_data = np_data_mat[:, batch_train_ind]
            train_data, trainlbl = reformat(train_data, trainlbl, num_class=clnum)
            train_data, trainlbl = randomize(train_data, trainlbl)
            num_tr_iter = int(train_data.shape[0] / batch_size)
            for iteration in range(num_tr_iter+1):
                global_step += 1
                start = iteration*batch_size
                end = (iteration+1)*batch_size
                train_batch, trainlbl_batch = get_next_batch(train_data, trainlbl, start=start, end=end)
                if trainlbl_batch.size == 0:
                    continue
                # Run optimization op (backprop)
                feed_dict_batch = {x: train_batch, lbl: trainlbl_batch, phase_train: True, keep_prob: keep_prob_value}
                if iteration == 104:
                    dd = 0

                # _, loss11 = sess.run([Optimizer, loss], feed_dict=feed_dict_batch)
                # if np.isnan(loss11):
                #     print("{0:3d}:".format(iteration), train_batch[1, 1:5, 1:5, :])
                # else:
                #     print("{0:3d}:".format(iteration), loss11)
                # ooo = tf.cond(tf.is_nan(loss11), lambda: x, lambda: loss11)
                _, accuracy_batch = sess.run([Optimizer, accuracy], feed_dict=feed_dict_batch)
                sum_acc_train = sum_acc_train + accuracy_batch
                counter_batch = counter_batch + 1
                if iteration % display_freq == 0:
                    # loss_batch, accuracy_batch, summery_tr = sess.run([loss, accuracy, merged], feed_dict_batch)
                    # summery_writer.add_summary(summery_tr, global_step)
                    loss_batch = sess.run(loss, feed_dict_batch)
                    print("iter {0:3d}:\t Loss={1:.2f},\tBatch Training Accuracy={2:.01%}".format(iteration, loss_batch,
                                                                                            accuracy_batch))
                    print("Accuracy={0:.01%}".format(sum_acc_train / counter_batch))
        print("Accuracy={0:.01%}".format(sum_acc_train/counter_batch))
        trainmodel_name = "model_dp{0:.2f}_ep{1:d}.ckpt".format(1 - keep_prob_value, epoch)
        save_path = save_result_path+trainmodel_name
        saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)
        with open(save_result_path+'TrainResult_model'+'.txt', mode='a') as fid:
            fid.writelines("Model Name {} \n".format(trainmodel_name))
            fid.writelines("Final LowLevel Accuracy: {0:.5f} \n".format(sum_acc_train/counter_batch))
            fid.writelines("\n\n\n")

Training = False
if Training == False:
    filename_base_model = "model_dp0.30_ep"
    START_RESTORE_MODEL = 9
    END_RESTORE_MODEL = 9
    print('Testing phase')
    for epi in range(START_RESTORE_MODEL, END_RESTORE_MODEL+1):
        model_name = save_result_path+"{0:s}{1:d}.ckpt".format(filename_base_model, epi)
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, "{0:s}".format(model_name))
            print("Model {0:s} is restored".format(model_name))
            sum_acc_test = 0
            counter_batch = 0
            miss = 0
            hit = 0
            confusion_matrix = np.zeros(shape=(n_classes, n_classes), dtype=np.int)
            for i in range(1, cntBatch + 1):
                all_prediction = np.array([], dtype=np.int32)
                all_outputs = np.empty(shape=(0, n_classes), dtype=np.float32)
                test_cl_ind = np.zeros(file_num, np.bool)
                data_path = expDir + 'data_resize' + str(fixedheight) + '_window' + str(windowsize) + '_step' + str(step - 1) + '_normal_' + fname + \
                            '_' + str(i) + '.mat'
                data_mat = h5py.File(data_path)
                np_data_mat = np.array(data_mat['data'])
                c22, rd = data_mat['data'].shape

                ss = int(k800 * nAug_per_file)
                if i <= cntBatch:
                    batch_test_ind = test_ind[(i - 1) * ss: i * ss]
                    if np.sum(batch_test_ind == True) == 0:
                        continue
                    test_cl_ind[(i - 1) * ss: i * ss] = batch_test_ind
                    testlbl = np_cl_mat[class_id, test_cl_ind]
                else:
                    batch_test_ind = test_ind[(i - 1) * ss:]
                    if np.sum(batch_test_ind == True) == 0:
                        continue
                    testlbl = np_cl_mat[class_id, test_cl_ind]

                r = np.sum(test_cl_ind == True)
                print('DataFile ' + str(i) + ' size = ' + str(rd) + ' Num test={}'.format(r))
                test_data = np_data_mat[:, batch_test_ind]

                # file_num = len(np.unique(np_cl_mat[new_fid, test_cl_ind]))
                # record_num = len(np.unique(np_cl_mat[file_id, test_cl_ind]))
                # clnum = len(np.unique(np_cl_mat[class_id, test_cl_ind]))

                test_data, testlbl = reformat(test_data, testlbl, num_class=clnum)
                num_tr_iter = int(test_data.shape[0] / test_batch_size)
                global_step = 0

                for iteration in range(num_tr_iter + 1):
                    global_step += 1
                    start = iteration * test_batch_size
                    end = (iteration + 1) * test_batch_size
                    test_batch, testlbl_batch = get_next_batch(test_data, testlbl, start=start, end=end)
                    if testlbl_batch.size == 0:
                        continue
                    # Run optimization op (backprop)
                    feed_dict_batch = {x: test_batch, lbl: testlbl_batch, phase_train: False, keep_prob: 1}
                    test_accuracy_batch, test_cls_prediction, test_output_logits = sess.run([accuracy, cls_prediction, output_logits], feed_dict_batch)
                    all_prediction = np.append(all_prediction, test_cls_prediction)
                    all_outputs = np.append(all_outputs, test_output_logits, axis=0)
                    sum_acc_test = sum_acc_test + test_accuracy_batch
                    counter_batch = counter_batch + 1
                    if iteration % display_freq == 0:
                        print("Accuracy={0:.01%}".format(sum_acc_test / counter_batch))
                        # print("test_output_logits={}".format(test_cls_prediction))

                ffiidd = np_cl_mat[file_id, test_cl_ind]
                _, iiddxx = np.unique(ffiidd, return_index=True)
                file_id_set = [ffiidd[index] for index in sorted(iiddxx)]
                set1 = np_cl_mat[:, test_cl_ind]
                st, en, ii = 0, 0, 0
                for id in file_id_set:
                    tempidx = set1[file_id, :] == id
                    true_class = set1[class_id, tempidx]
                    en = st + int(nAug_per_file)
                    out1 = all_outputs[st:en, :]
                    st = en
                    ii += 1
                    win_class = np.unravel_index(out1.argmax(), out1.shape)[1]
                    confusion_matrix[int(true_class[0]-1), int(win_class)] += 1
                    if int(win_class) == int(true_class[0]-1):
                        hit += 1
                    else:
                        miss += 1
            final_acc = hit/(miss+hit)
        with open(save_result_path+'Result_model'+'.txt', mode='a') as fid:
            fid.writelines("Model Name {} \n".format(model_name))
            fid.writelines("Final Accuracy: {0:.5f} \n".format(final_acc))
            fid.writelines("Confusion Matrix=\n {}\n".format(confusion_matrix))
            fid.writelines("\n\n\n")



