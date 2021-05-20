import argparse
import os
import sys
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import provider
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 65536]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a image')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 23

#需要设置
DATA_PATH = '../dataset/'

#默认用05序列做预测
#提取要预测的文件的序列号 如000001
RANK_LIST = [line.rstrip()[:-4] for line in open(DATA_PATH+'/sequence/05/velodyne/all_file_greater_65536.txt')]

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}

    total_correct = 0
    total_seen = 0

    #输出文件
    fout_out_filelist = open(FLAGS.output_filelist,'w')
    for line in RANK_LIST:
        out_data_label_filename = line+'_pred.txt'
        out_data_label_filename = os.path.join(DUMP_DIR,out_data_label_filename)

        out_true_label_filename = line + '_true.txt'
        out_true_label_filename = os.path.join(DUMP_DIR, out_true_label_filename)
        print(line, out_data_label_filename)

        a, b = eval_one_epoch(sess, ops, line, out_data_label_filename, out_true_label_filename)
        total_correct += a
        total_seen += b

        #输出结果
        fout_out_filelist.write(out_data_label_filename+'\n')
    fout_out_filelist.close()
    log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))

#每轮检测只投入一帧(one image)
def eval_one_epoch(sess, ops, rank, out_data_label_filename, out_true_label_filename):
    is_training = False
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    #打开要输出到的文件
    fout_data_label = open(out_data_label_filename, 'w')
    fout_true_label = open(out_true_label_filename, 'w')

    #读入点的信息 这里维数先用二维的 可能有Bug
    current_data, current_label = provider.load_one_image(DATA_PATH,rank)
    #current_data = current_data.reshape((1,NUM_POINT,3))
    #current_label = current_label.reshape((1,NUM_POINT))

    feed_dict = {ops['pointclouds_pl']: current_data,
                 ops['labels_pl']: current_label,
                 ops['is_training_pl']: is_training}
    loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                  feed_dict=feed_dict)

    # If true, do not count the clutter class
    pred_label = np.argmax(pred_val, 1)  # N
    for i in range(len(pred_label)):
        fout_data_label.write("%f %f %f %f %d\n" % (current_data[i][0],current_data[i][1],current_data[i][2],pred_val[i,pred_label[i]],pred_label[i]))
        fout_true_label.write("%d\n"%(current_label[i]))

    currect = np.sum(pred_label == current_label)
    seen = NUM_POINT
    loss = loss_val
    for i in range(NUM_POINT):
        l = current_label[i]
        total_seen_class[l] += 1
        total_correct_class += 1 if pred_label[i] == l else 0

    log_string('eval loss: %f' % (loss))
    log_string('eval accuracy: %f'% (currect/seen))
    fout_data_label.close()
    fout_true_label.close()