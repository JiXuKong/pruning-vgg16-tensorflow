import os 
import tensorflow as tf 
import cv2
import numpy as np
import config as cfg

'''
turn data into tfrecords
The data path form:
...datapath
......class_path
.........img_path
'''


if __name__ == '__main__':
    datapath = '' 
    savepath = ''
    classes=['class1','class2', 'class3', 'class4']
    writer= tf.python_io.TFRecordWriter(os.path.join(savepath, "train.tfrecords"))
    img_path_ = []
    for i in range(len(classes)):
        index = i
        name = classes[i]
        class_path = os.path.join(datapath, name)
        for img_name in (os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            img_path_.append([img_path,index])

    np.random.shuffle(img_path_)
    for i in range(len(img_path_)):
        index = img_path_[i][1]
        img_path = img_path_[i][0]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        mean = [np.mean(img[:,:,i]) for i in range(3)]
        mean = np.array(mean).reshape(1,1,3)
        std = [np.std(img[:,:,i]) for i in range(3)]
        std = np.array(std).reshape(1,1,3)
        img = (img - mean)/(std + 1e-30)
        img_raw=img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()
    print('Successfully write to: ', savepath)    
    
