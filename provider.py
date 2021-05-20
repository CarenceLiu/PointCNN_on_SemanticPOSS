import numpy as np
import os

def read_points(bin_file):
    '''
    bin_file：文件路径
    读取二进制点云文件 以ndarray形式返回 points.shape = (n,4) 其中n为点的个数
    '''
    points = np.fromfile(bin_file, dtype=np.float32)
    points = np.reshape(points, (-1, 4))  # x,y,z,intensity
    return points


def read_semlabels(label_file):
    semlabels = np.fromfile(label_file, dtype=np.uint32) & 0xffff
    return semlabels


def read_inslabels(label_file):
    inslabels = np.fromfile(label_file, dtype=np.uint32) >> 16
    return inslabels

def load_one_sequence(dataset_path,rank):
    '''
    dataset_path为semanticPOSS的dataset的路径
    rank为读入sequence的序号 输入0-5 对应sequences中00-05
    读取all_file_greater_65536 默认返回是一个n*65536*3的ndarray对象与n*65536的ndarray对象组成的元组，分别为点信息与label信息
    '''
    data_path_velodyne = dataset_path+'/sequences/0' + str(rank)+'/velodyne/'
    data_path_labels = dataset_path+'/sequences/0' + str(rank)+'/labels/'
    ALL_FILE_VELODYNE = open(data_path_velodyne+'all_file_greater_65536.txt','r')
    ALL_FILE_LABELS = open(data_path_labels+'all_file_greater_65536.txt','r')
    points_list = []
    labels_list = []
    for file in ALL_FILE_VELODYNE:
        file = file.rstrip()
        points = read_points(data_path_velodyne+file)[:65536,0:3]
        points_list.append(points)
    for file in ALL_FILE_LABELS:
        file = file.rstrip()
        labels = read_semlabels(data_path_labels+file)[:65536]
        labels_list.append(labels)
    return  (np.array(points_list),np.array(labels_list))

def load_one_image(dataset_path,file_rank):
    '''
    dataset_path为dataset的路径
    file_rank为预测集中的文件序号 如000001
    返回为(data,label)的元组 data为ndarray对象 (65536*3) label为ndarray对象 (65536)
    '''
    data_path= dataset_path+'/sequences/05/velodyne/'+file_rank+'.bin'
    label_path= dataset_path+'/sequences/05/labels/'+file_rank+'.label'
    data = read_points(data_path)[:65536,:3]
    label = read_semlabels(label_path)[:65536]
    return (data,label)

def init_dataset(dataset_path):
    '''
    在初始化的时候运行文件 以创建all_file.txt以及all_file_65536.txt
    all_file.txt中为所有数据集的文件名
    all_file_65536.txt为点个数大于65536的数据集文件名
    '''
    for i in range(6):
        #此处路径可能要做对应替换
        data_path = dataset_path+'/sequences/0'+str(i)
        data_list_65536 = open(data_path+'/velodyne/all_file_greater_65536.txt','w')
        data_list = open(data_path+'/velodyne/all_file.txt','w')
        label_list_65536 = open(data_path+'/labels/all_file_greater_65536.txt','w')
        label_list = open(data_path + '/labels/all_file.txt', 'w')
        for file_name in sorted(os.listdir(data_path+'/velodyne/')):
            if file_name.endswith(".bin"):
                data_list.write(file_name + '\n')
                if read_points(data_path+'/velodyne/'+file_name).shape[0] >= 65536:
                    data_list_65536.write(file_name+'\n')
        for file_name in sorted(os.listdir(data_path+'/labels/')):
            if file_name.endswith(".label"):
                label_list.write(file_name + '\n')
                if read_semlabels(data_path+'/labels/'+file_name).shape[0] >= 65536:
                    label_list_65536.write(file_name+'\n')
        data_list.close()
        label_list.close()
        data_list_65536.close()
        label_list_65536.close()


#测试
#points_, labels_ = load_one_image('../dataset/',"000001")
#print(np.squeeze(labels_).shape)
