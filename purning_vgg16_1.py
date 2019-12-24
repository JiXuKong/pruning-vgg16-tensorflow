import numpy as np
import tensorflow as tf
import os
from vgg import vgg_16
import config as cfg
from timer import Timer
import datetime
import math
slim = tf.contrib.slim

def load_data(train_data_path):
    filename_queue = tf.train.string_input_producer([train_data_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])
    label = tf.reshape(tf.cast(features['label'], tf.int64), [-1,])
    return img, label


def network(x):
    net, net_summary = vgg_16(inputs = x, num_classes = cfg.CLASSES1, is_training = True)
    net = tf.reshape(net, [-1, cfg.CLASSES1])
    return net

def label_encode(labels):
    labels = tf.reshape(labels,[-1,])
    labels = tf.cast(labels,dtype=tf.int32)
    one_hot = tf.one_hot(labels, cfg.CLASSES1)
    return one_hot

def evaluation(pred_list, y_true):
    with tf.name_scope("evaluation"):
        accu_1, rec_1, prec_1, f1_1 = [], [], [], []
        pred_list = tf.cast(tf.argmax(pred_list, 1),tf.int32)
        pred_list = tf.reshape(pred_list, [-1, 1])
        y_true = tf.cast(y_true, tf.int32)
        tp_record, tn_record, fp_record, fn_record = {}, {}, {}, {}
        
        for i in range(cfg.CLASSES1):
            temp = tf.ones(tf.shape(pred_list), tf.int32)
            pred_find_iindex = tf.where(tf.equal(pred_list, i*temp), temp, 0*temp)
            y_t_find_iindex = tf.where(tf.equal(y_true, i*temp), temp, 0*temp)
            temp_add = tf.add(pred_find_iindex, y_t_find_iindex)
            temp_subtract = tf.subtract(pred_find_iindex, y_t_find_iindex)

            tp = tf.cast(tf.reduce_sum(tf.cast(tf.equal(temp_add, 2), tf.int32)), tf.float32)
            tn = tf.cast(tf.reduce_sum(tf.cast(tf.equal(temp_add, 0), tf.int32)), tf.float32)
            fp = tf.cast(tf.reduce_sum(tf.cast(tf.equal(temp_subtract, 1), tf.int32)), tf.float32)
            fn = tf.cast(tf.reduce_sum(tf.cast(tf.equal(temp_subtract, -1), tf.int32)), tf.float32)

            if (len(tp_record) < cfg.CLASSES1):
                tp_record[str(i)] = tf.add(tp, 1e-30)
                tn_record[str(i)] = tf.add(tn, 1e-30)
                fp_record[str(i)] = tf.add(fp, 1e-30)
                fn_record[str(i)] = tf.add(fn, 1e-30)
            else:
                tp_record[str(i)] = tp + tp_record[str(i)]
                tn_record[str(i)] = tn + tn_record[str(i)]
                fp_record[str(i)] = fp + fp_record[str(i)]
                fn_record[str(i)] = fn + fn_record[str(i)]

            if not cfg.ACCUMULATION:
                tp1 = tp
                tn1 = tn
                fp1 = fp
                fn1 = fn
            else:
                tp1 = tp_record[str(i)]
                tn1 = tn_record[str(i)]
                fp1 = fp_record[str(i)]
                fn1 = fn_record[str(i)]

            tp_tn1 = tf.add(tp1, tn1)
            fp_fn1 = tf.add(fp1, fn1)
            accuracy1 = tf.divide(tp_tn1, tf.add(tp_tn1, fp_fn1))
            recall1 = tf.divide(tp1, tf.add(tf.add(tp1, fn1), 1e-30))
            precision1 = tf.divide(tp1, tf.add(tf.add(tp1, fp1), 1e-30))
            p_r_multi1 = tf.multiply(precision1, recall1)
            F1_score1 = tf.divide(tf.multiply(2.0, p_r_multi1), tf.add(tf.add(precision1, recall1), 1e-30))

            accu_1.append(accuracy1)
            rec_1.append(recall1)
            prec_1.append(precision1)
            f1_1.append(F1_score1)
    return accu_1, rec_1, prec_1, f1_1        


def evaluate(pred, labels):
    print(pred.get_shape().as_list())
    softmax = tf.nn.softmax(pred, axis = 1)
    accu, rec, prec, f1 = evaluation(softmax, labels)
    return accu, rec, prec, f1


def cross_softmax(logits, label):
    one_hot = label_encode(label)
    softmax = tf.nn.softmax(logits, axis = 1)
    cross_entropy_softmax = tf.reduce_mean(-tf.reduce_sum(one_hot*tf.log(softmax + 1e-30), -1))
    return cross_entropy_softmax


def load_weight(weight_path):
    reader = tf.train.NewCheckpointReader(weight_path)
    return reader

def purning(reader, purning_step, layer_sparsity):
    all_variables = reader.get_variable_to_shape_map()
    for key_weight in all_variables:
        if (key_weight.split('/')[-1] == 'weights')and(key_weight != 'vgg_16/fc8/weights'):#and(key_weight != 'vgg_16/fc7/weights')and(key_weight != 'vgg_16/fc6/weights'):
            print('Purning Weight: ', key_weight)
            key_biases = key_weight.split('weights')[0] + 'biases'
            print('Purning Biases: ', key_biases)
            w_weight = reader.get_tensor(key_weight)
            w_biases = reader.get_tensor(key_biases)
            w = w_weight + w_biases.reshape(1,1,1,w_biases.shape[0])
            w_index = np.arange(w.shape[-1])
        
            l1_w = np.sum(abs(w), axis = (0,1,2))
            l1_w_sort_index = np.argsort(l1_w)
            l1_w_sort = sorted(l1_w)
            current_left_sparsity = l1_w.shape[0] - int(l1_w.shape[0]*layer_sparsity[purning_step])
            max_left_sparsity =  32
            if current_left_sparsity > max_left_sparsity:
                sparsity = int(l1_w.shape[0]*layer_sparsity[purning_step])
            else:
                sparsity = max_left_sparsity
            print(l1_w.shape[0], layer_sparsity[purning_step])
#             sparsity = int(layer_sparsity[purning_step]*l1_w.shape[0])
            print(sparsity)
            purning_index = l1_w_sort_index[:sparsity]
            for j in range(purning_index.shape[0]):
                w_weight[:,:,:,purning_index[j]] = 0
                w_biases[purning_index[j]] = 0
            all_variables[key_weight] = w_weight
            all_variables[key_biases] = w_biases
    return all_variables

def _zero(all_variables):
    zero_num = 0
    all_num = 0
    for key_weight in all_variables:
        if (key_weight.split('/')[-1] == 'weights'):
            key_biases = key_weight.split('weights')[0] + 'biases'
            w_weight = np.array(all_variables[key_weight])
            w_biases = np.array(all_variables[key_biases])
            w = w_weight + w_biases.reshape(1,1,1,w_biases.shape[0])
            l1_w = np.sum(abs(w), axis = (0,1,2))

            print(key_weight, l1_w.shape[0], np.where(l1_w == 0)[0].shape[0])

            
            zero_num += np.where(w == 0)[0].shape[0]
            all_num += (w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3] + w.shape[3])
    sparsity = zero_num/all_num
    return sparsity, all_num, zero_num


def assign_to_varible(sess, all_variables):
    print('restore weights')
    for key in all_variables:
        if (key != 'vgg_16/fc8/weights')and(key != 'vgg_16/fc8/biases'):#and\
#         (key != 'vgg_16/fc7/weights')and(key != 'vgg_16/fc7/biases')and\
#         (key != 'vgg_16/fc6/weights')and(key != 'vgg_16/fc6/biases'):
            if key.split('/')[-1] == 'weights':
                with tf.variable_scope(key.split('/weights')[0], reuse = True):
                    print('restore: ', key)
                    sess.run(tf.get_variable('weights', trainable = True).assign(all_variables[key]))
                    w = tf.get_variable('weights').eval(session=sess)
                    print(np.where(np.sum(abs(w), axis = (0,1,2))==0)[0].shape)
            if key.split('/')[-1] == 'biases':
                with tf.variable_scope(key.split('/biases')[0], reuse = True):
                    print('restore: ', key)
                    sess.run(tf.get_variable('biases', trainable = True).assign(all_variables[key]))
        
        
def print_infor(accu, rec, prec, f1):
    for i in range(cfg.CLASSES1):
        string = 'The ' + cfg.CLASSES[i] + ' is {:5.3f}, recall is {:5.3f}, precision is {:5.3f}, f1 score is {:5.3f}'
        print(string.format(accu[i],rec[i],prec[i],f1[i]))    
            
 
    
def Run_record(sess, phase, run_op, step, feed_dict, purning_sparsity):
    mf1 = 0
    if (step == 0)or(step == 1):
        if step == 0:
            print('Step 0 Without Purning, The Test Performance: ')
        if step == 1:
            print('Step 1 First Purning, The Test Performance: ')
        loss, accu, rec, prec, f1 = sess.run(run_op, feed_dict = feed_dict)
        print_infor(accu, rec, prec, f1)
    else:
        if phase == 'Test':
            loss, accu, rec, prec, f1 = sess.run(run_op, feed_dict = feed_dict)
            print('Step ' + str(step) + 'Purning Sparsity: ' + str(purning_sparsity) + ' The Test Performance: ')
            print_infor(accu, rec, prec, f1)
            mf1 = sum(f1)/cfg.CLASSES1
        if phase == 'Train':
            _, loss, accu, rec, prec, f1 = sess.run(run_op, feed_dict = feed_dict)
            print('Step: ' + str(step) + ' Purning Sparsity: ' + str(purning_sparsity) + ' train loss: ', loss)
    writer.add_summary(tf.Summary(value=
                                  [tf.Summary.Value(tag="Purning" + phase + "Loss", simple_value=loss)]), step)    
    for i in range(cfg.CLASSES1):
        writer.add_summary(tf.Summary(value=
                                  [tf.Summary.Value(tag="Purning" + phase + "Performance/" 
                                                     + cfg.CLASSES[i] + " f1 score", simple_value=f1[i])]),step)
                                  
                                  
                                  
                                 
    return mf1






if __name__ == '__main__':


    train_data_path = 'tfrecords/train.tfrecords'
    test_data_path = 'tfrecords/test.tfrecords'
    init_weight_path = 'weight/...ckpt'


    '''hyper_parameters'''
    init_learning_rate = 1e-5
    
    layer_sparsity = np.array([0.0496,0.22,0.324,0.35,0.48,0.506,0.518,0.55,0.585,0.595,
                        0.629,0.65,0.68,0.707,0.736,0.765,0.77,0.78,0.786,0.79,0.80,0.805,0.81,0.815,                              0.82,0.825,0.83,0.835,0.84,0.845,0.85,0.85,0.86,0.865,0.87,0.875,0.88,0.885,0.89,0.895,0.90,0.905,0.91,0.915,0.92,0.925,0.93])
    purning_step = 200


    x = tf.placeholder(tf.float32, [None, cfg.IMAGE_SIZE,  cfg.IMAGE_SIZE, 3],name='images')
    y = tf.placeholder(tf.float32,[None, 1])


    net = network(x)
    cross_entropy_softmax = cross_softmax(net, y)
    accu_1, rec_1, prec_1, f1_1 = evaluate(net, y)


    output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ckpt_file = os.path.join(output_dir, 'purning_vgg16')


    train_img, train_label = load_data(train_data_path)
    test_img, test_label = load_data(test_data_path)

    with tf.device('/cpu:0'):
        train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img,train_label],batch_size=cfg.BATCH_SIZE,num_threads=3, capacity=2600,min_after_dequeue=1000)
        test_img_batch, test_label_batch = tf.train.batch([test_img, test_label], batch_size=cfg.BATCH_SIZE, num_threads=3, capacity=64,allow_smaller_final_batch=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=32)
    saver1 = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    writer = tf.summary.FileWriter(output_dir, flush_secs=100)
    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate)
    train_op = slim.learning.create_train_op(cross_entropy_softmax, optimizer, global_step=global_step)

    sess = tf.Session()
    sess.run(tf.variables_initializer(tf.global_variables(), name='init'))
    if init_weight_path is not None:
                print('Restoring weights from: ' + init_weight_path)
                saver1.restore(sess, init_weight_path)

    coord=tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)


    best_f1 = 0
    images1, labels1 = sess.run([test_img_batch, test_label_batch])
    for step in range(0, 2000):
  
        images, labels = sess.run([train_img_batch, train_label_batch])
        if (step == 0)or(step == 1)or (step%purning_step == 0)or(step%21 == 0):
            if step == 0:
                feed_dict = {x: images,
                             y: labels}
                run_op = [cross_entropy_softmax, accu_1, rec_1, prec_1, f1_1]
                _ = Run_record(sess, 'Test', run_op, step, feed_dict, 0)
            if (step == 1):
                weight_reader = load_weight(init_weight_path)
                all_variables = purning(weight_reader, 0, layer_sparsity)
                sparsity, all_num, zero_num = _zero(all_variables)
                print('sparsity: ', sparsity)
                assign_to_varible(sess, all_variables)

                feed_dict1 = {x: images1,
                             y: labels1}

                run_op1 = [cross_entropy_softmax, accu_1, rec_1, prec_1, f1_1]
                _ = Run_record(sess, 'Test', run_op1, step, feed_dict1, 1)

            if (step%purning_step == 0)and(step != 0):
                coord.request_stop()
                coord.join(threads)
                sess.close()#关闭会话

                sess = tf.Session()#重开会话
                sess.run(tf.variables_initializer(tf.global_variables(), name='init'))#重新初始化

                coord=tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                images1, labels1 = sess.run([test_img_batch, test_label_batch])
                images, labels = sess.run([train_img_batch, train_label_batch])
                
                
                best_f1 = 0

                print('Restoring weights from: ' + ckpt_file + '_Sparsity_' + str(step//purning_step-1) + '_best_performance-0')
                saver1.restore(sess, ckpt_file + '_Sparsity_' + str(step//purning_step-1) + '_best_performance-0')

                weight_reader = load_weight(ckpt_file + '_Sparsity_' + str(step//purning_step-1) + '_best_performance-0')
                all_variables = purning(weight_reader, int(step//purning_step), layer_sparsity)
                sparsity, all_num, zero_num = _zero(all_variables)
                print('sparsity: ', sparsity)
                assign_to_varible(sess, all_variables)

                feed_dict1 = {x: images1,
                             y: labels1}
                _ = run_op1 = [cross_entropy_softmax, accu_1, rec_1, prec_1, f1_1]
                Run_record(sess, 'Test', run_op1, step, feed_dict1, layer_sparsity[int(step//purning_step)])
            if step%21 == 0:
                feed_dict1 = {x: images1,
                             y: labels1}
                run_op1 = [cross_entropy_softmax, accu_1, rec_1, prec_1, f1_1]
                mf1 = Run_record(sess, 'Test', run_op1, step, feed_dict1, layer_sparsity[int(step//purning_step)])
                if mf1 > best_f1:
                    best_f1 = mf1
                    print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    output_dir))
                    saver.save(sess, ckpt_file + '_Sparsity_' + str(step//purning_step) + '_best_performance', global_step = 0)
        else:
            feed_dict = {x: images,
                             y: labels}
            run_op = [train_op, cross_entropy_softmax, accu_1, rec_1, prec_1, f1_1]
            
            Run_record(sess, 'Train', run_op, step, feed_dict, layer_sparsity[int(step//purning_step)])
#             with tf.variable_scope('vgg_16/conv1/conv1_1', reuse = True):
#                 w1 = tf.get_variable('weights').eval(session=sess)
#                 print(np.where(np.sum(abs(w1), axis = (0,1,2))==0)[0].shape)
            
    coord.request_stop()
    coord.join(threads)
