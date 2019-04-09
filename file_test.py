import tensorflow as tf
import argparse
import cv2
import numpy as np
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def setup_inference_graph(path_to_pb):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_label_map(path_to_pbtxt):
    with open(path_to_pbtxt) as f_in:
        pbtxt = ''.join(f_in.readlines())
    num_classes=pbtxt.count('id')
    label_map = label_map_util.load_labelmap(path_to_pbtxt)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load an inference graph and use it on a webcam stream")
    parser.add_argument('-g','--graph',help="path to the pb file with the graph and weight definition",required=True)
    parser.add_argument('-l','--labelMap',help="path to the pbtxt containing the label definition",required=True)
    parser.add_argument('-f','--files',help="files where we want to visualize predicitons",required=True, nargs='+')
    parser.add_argument('-th',help='minimum threshold to display predictions',type=float, default=0.5)
    args = parser.parse_args()

    for path in [args.graph,args.labelMap]+args.files:
        if not os.path.exists(path):
            print('ERROR: Unable to find {}'.format(path))
            exit()
    
    print('setting up graph')
    detection_graph = setup_inference_graph(args.graph)

    print('Load image labels')
    category_index = load_label_map(args.labelMap)

    #setting up tensorflow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            #fetch usefull stuff
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            cv2.namedWindow('detection',cv2.WINDOW_NORMAL)
            for f in args.files:
                #if f.endswith('.')
                read_frame = cv2.imread(f, -1)

                #convert bgr -> rgb
                #convert bgr to rgb
                read_frame = read_frame[:,:,::-1]

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                img = np.expand_dims(read_frame,axis=0)


                # Actual detection.
                (bs, ss, cs, ns) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: img})
                
                #draw detection
                vis_util.visualize_boxes_and_labels_on_image_array(
                        read_frame,
                        np.squeeze(bs),
                        np.squeeze(cs).astype(np.int32),
                        np.squeeze(ss),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=args.th)
                #back to bgr
                read_frame=read_frame[:,:,::-1]

                #show result
                cv2.imshow('detection', read_frame)
                cv2.waitKey(0)