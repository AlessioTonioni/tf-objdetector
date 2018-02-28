import tensorflow as tf
import time
import argparse
import cv2
import numpy as np
import os
import threading
import queue
from object_detection.utils import label_map_util
from random import randint

def setup_inference_graph(path_to_pb):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def random_color():
    return (randint(0,255),randint(0,255),randint(0,255))

def cv_draw_box(box,score,c_id,class_names,image,line_thickness=8,font_scale=1):
    ymin, xmin, ymax, xmax = box
    ymin_abs = int(ymin*image.shape[0])
    ymax_abs = int(ymax*image.shape[0])
    xmin_abs = int(xmin*image.shape[1])
    xmax_abs = int(xmax*image.shape[1])

    class_id = int(c_id)
    text_to_display = "{} - score:{:.2f}".format(class_names[class_id][0],score)
    cv2.rectangle(image,(xmin_abs,ymin_abs),(xmax_abs,ymax_abs),class_names[class_id][1],line_thickness)
    cv2.putText(image,text_to_display,(xmin_abs,ymin_abs),cv2.FONT_HERSHEY_SIMPLEX,font_scale,255,2)
    return image

def cv_draw_detection(boxes,scores,classes,classes_names,image,conf_th=0.1,line_thickness=8,font_scale=1):
    for (b,s,c) in zip(boxes,scores,classes):
        if s>conf_th:
            image=cv_draw_box(b,s,c,classes_names,image,line_thickness=line_thickness,font_scale=font_scale)
    return image


camera_frames = queue.Queue(1)

class ImageGrabber(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.ID=ID
        self.cam=cv2.VideoCapture(ID)

    def run(self):
        global camera_frames
        while True:
            ret,frame=self.cam.read()
            camera_frames.put(frame)
            time.sleep(0.02)

class Main(threading.Thread):
    def __init__(self,args):
        print(args)
        self.args=args
        threading.Thread.__init__(self)

    def run(self):
        global camera_frames
        print('setting up graph')
        detection_graph = setup_inference_graph(self.args.graph)

        #setting up tensorflow session
        with detection_graph.as_default():
            #print('\n'.join([n.name for n in detection_graph.as_graph_def().node]))
            #exit()
            # FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Conv2D
            layer_name = 'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0'
            #layer_name = 'FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D:0'

            #10x10x1024 feature map
            features = detection_graph.get_tensor_by_name(layer_name)

            #actual detection [N,4] array with each row being ymin, xmin, ymax, xmax = box
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            boxes = tf.squeeze(boxes)

            #create a descriptors db db
            features_dimension = tf.shape(features)
            static_feature_dimension = features.get_shape()
            embedding_db = tf.Variable(tf.zeros(shape=[1,static_feature_dimension[-1]]),name='embedding_db')

            #fetch embeddings
            tiled_dimension = tf.cast(tf.tile(features_dimension[1:3],[2]),dtype=tf.float32)
            absolute_coord = tf.cast(tf.floor(boxes*tiled_dimension),dtype=tf.int32)
            detected_embeddings = []
            classes_list = []

            def extract_descriptor_and_class(coordinates):
                box_feature = features[:,coordinates[0]:coordinates[2],coordinates[1]:coordinates[3],:]
                pooled_dimension = tf.shape(box_feature)
                mac_embedding = tf.reduce_max(box_feature,axis=[1,2])
                normalized_mac_embeddings = tf.nn.l2_normalize(mac_embedding,axis=-1)
                cosine_similarity = tf.reduce_sum(normalized_mac_embeddings*embedding_db,axis=-1)
                class_prediction = tf.argmax(cosine_similarity)
                class_prediction = tf.cast(class_prediction,dtype=tf.float32)
                return class_prediction,normalized_mac_embeddings
            
            classes,detected_embeddings = tf.map_fn(extract_descriptor_and_class,absolute_coord,back_prop=False, dtype=(tf.float32, tf.float32))
            classes = tf.squeeze(classes,name='class_prediction')
            detected_embeddings = tf.squeeze(detected_embeddings)
            

            # #print(detection_graph.get_tensor_by_name('Conv2d_13_pointwise:0'))
            with tf.Session(graph=detection_graph) as sess:
                sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
                #fetch usefull stuff
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = tf.squeeze(detection_graph.get_tensor_by_name('detection_scores:0'))
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                cv2.namedWindow('my webcam',cv2.WINDOW_NORMAL)

                step1=True
                class_names = []
                first_call=True
                Stop=True
                print('Press Enter to acquire new class, s to save')
                while Stop:
                    if camera_frames.empty():
                        continue
                    else:
                        read_frame=camera_frames.get()

                    cv2.imshow('my webcam', read_frame)
                    result = cv2.waitKey(1)
                    print(result)

                    #convert bgr to rgb
                    read_frame = read_frame[:,:,::-1]

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    img = np.expand_dims(read_frame,axis=0)
                    
                    if (step1 and (result==141 or result==13)) or (not step1 and result!=115):
                        if step1:
                            #Object Acgnostic Detection
                            (bs,ss,ns,ee) = sess.run([boxes, scores, num_detections, detected_embeddings],feed_dict={image_tensor: img})

                            cs = np.zeros_like(ss)

                            #back to bgr
                            read_frame=read_frame[:,:,::-1]

                            #check if there is a good enough box to save it's embedding
                            for b,s,e in zip(bs,ss,ee):
                                if s>self.args.th:
                                    to_display = np.copy(read_frame)
                                    cv_draw_box(b,s,0,[('object',random_color())],to_display)

                                    cv2.imshow('Candidate', to_display)
                                    cv2.waitKey(0)
                                    
                                    answer = input('Would you like to add a new descriptor?[Y/N]')
                                    while answer not in ['Y','N','y','n']:
                                        print('Unrecognized sequence')
                                        answer = input('Would you like to add a new descriptor?[Y/N]')
                                    if answer=='N' or answer=='n':
                                        continue
                                    else:
                                        #retrieve current embedding
                                        emby = sess.run(embedding_db)
                                        e = np.expand_dims(e,axis=0)
                                        if  not first_call:
                                            emby = np.concatenate([emby,e])
                                        else:
                                            emby = e
                                            first_call=False

                                        #change embedding_db
                                        sess.run(tf.assign(embedding_db,emby,validate_shape=False))

                                        #add class name
                                        class_name = input('Class Name:')
                                        color = random_color()
                                        for n,c in class_names:
                                            if n == class_name:
                                                color=c
                                                break
                                        class_names.append((class_name,color))
                            if len(class_names)>0:
                                answer = input('Add other Classes?')
                                step1 = (answer=='Y' or answer=='y')
                        else:
                            # Actual detection.
                            (bs, ss, cs, ns,ee) = sess.run([boxes, scores, classes, num_detections,detected_embeddings],feed_dict={image_tensor: img})

                            
                            #back to bgr
                            read_frame=read_frame[:,:,::-1]

                            read_frame=cv_draw_detection(bs,ss,cs,class_names,read_frame,conf_th=self.args.th)

                            #show result
                            cv2.imshow('my webcam', read_frame)
                            cv2.waitKey(1)            
                    
                    elif result==115:
                        #convert db to constant and export graph
                        graph_1_pb = detection_graph.as_graph_def()
                        graph_2_pb = tf.graph_util.convert_variables_to_constants(sess,graph_1_pb,['embedding_db','image_tensor','detection_scores','num_detections','detection_boxes','class_prediction'])
                        # Finally we serialize and dump the output graph to the filesystem
                        with tf.gfile.GFile('detection_graph.pb', "wb") as f:
                            f.write(graph_2_pb.SerializeToString())

                        #convert classnames to labellist
                        to_write=[]
                        for i,c in enumerate(class_names):
                            #to_write.append('{\n\tid:'+str(i)+'\n\tname:'+c[0]+'\n}\n')
                            to_write.append('{}\n'.format(c[0]))
                        with open('label_map.txt','w+') as f_out:
                            f_out.writelines(to_write)
                        print('All Done bye bye')
                        Stop=False

                    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load an inference graph and use it on a webcam stream")
    parser.add_argument('-g','--graph',help="path to the pb file with the graph and weight definition",required=True)
    parser.add_argument('-th',help='confidence threshold to show detection',type=float,default=0.3)
    parser.add_argument('-c','--camera',help="camera number to be used", default=0, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        print('ERROR: Unable to find {}'.format(args.graph))
        exit()
    grabber = ImageGrabber(args.camera)
    main = Main(args)

    grabber.start()
    main.start()
    main.join()
    grabber.join()

    # print('setting up graph')
    # detection_graph = setup_inference_graph(args.graph)

    # print(detection_graph.get_tensor_by_name('class_prediction:0'))

    