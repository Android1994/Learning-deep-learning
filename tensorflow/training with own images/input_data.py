# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

def get_files(file_dir):  ###########################################################
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    classes={'edge','noedge'} #2 class
    
    edges = []
    label_edges = []
    noedges = []
    label_noedges = []
    
    for index, name in enumerate(classes):
        class_path=file_dir+name+'/'
        for img_name in os.listdir(class_path): 
            img_path=class_path+img_name #image path
            if name=='edge':
                edges.append(img_path)
                label_edges.append(1)
            else:
                noedges.append(img_path)
                label_noedges.append(0)
    print('There are %d edges\nThere are %d noedges' %(len(edges), len(noedges)))
    
    image_list = np.hstack((edges, noedges))
    label_list = np.hstack((label_edges, label_noedges))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    return image_list, label_list


def get_batch(image, label, BATCH_SIZE, CAPACITY, IMG_W, IMG_H):  ####################################
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=1) ##############
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, IMG_W, IMG_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)
    
#    image_batch, label_batch = tf.train.batch([image, label],
#                                              batch_size= 50,
#                                                num_threads= 64, 
#                                                capacity = 2000)
    
    #you can also use shuffle_batch 
    image_batch, label_batch = tf.train.shuffle_batch([image,label],
                                                      batch_size=BATCH_SIZE,
                                                      num_threads=64,
                                                      capacity=CAPACITY,
                                                      min_after_dequeue=CAPACITY-3*BATCH_SIZE
                                                     )
    label_batch = tf.reshape(label_batch, [BATCH_SIZE])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch
