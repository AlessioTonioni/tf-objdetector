import tensorflow as tf
import argparse
import os
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
from matplotlib import pyplot as plt
import time

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


def predict(test_image_paths, detection_graph, category_index=None, visualization=False):
    result= [{} for _ in range(len(test_image_paths))]
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

            for index,image_path in enumerate(test_image_paths):
                #laod images
                s_t = time.clock()
                image_np = np.array(Image.open(image_path))

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                (bs, ss, cs, ns) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
   
                result[index]['boxes']=bs
                result[index]['scores']=ss
                result[index]['classes']=cs
                result[index]['num_detections']=ns

                print('{}/{}'.format(index,len(test_image_paths)),end='\r')

                if visualization:
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(bs),
                        np.squeeze(cs).astype(np.int32),
                        np.squeeze(ss),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    plt.figure()
                    plt.imshow(image_np)
                    plt.show()
    
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load an inference graph and use it on the input images")
    parser.add_argument('-g','--graph',help="path to the pb file with the graph and weight definition",required=True)
    parser.add_argument('-l','--labelMap',help="path to the pbtxt containing the label definition",required=True)
    parser.add_argument('-t','--target',help="path to an image to test or to a txt file with the list of image to test (one path per row)",required=True)
    parser.add_argument('-v','--visualization',help="flag to enable visualization",action='store_true')
    parser.add_argument('-o','--output',help="output folder were the detections will be saved",required=True)
    parser.add_argument('-b','--base_folder',help="base folder for the test image",default='')
    args = parser.parse_args()

    for path in [args.graph,args.labelMap,args.target]:
        if not os.path.exists(path):
            print('ERROR: Unable to find {}'.format(path))
            exit()
    
    if args.target.endswith('.txt'):
        print('Going to use test file in yolo format')
        with open(args.target) as f_in:
            img_to_test=[p.strip() for p in f_in.readlines()]
        print('Image to test: {}'.format(len(img_to_test)))
    else:
        print('Detection on single image file: {}'.format(args.target))
        img_to_test=[args.target]
    
    print('setting up graph')
    detection_graph = setup_inference_graph(args.graph)

    print('Load image labels')
    category_index = load_label_map(args.labelMap)

    print('Start detection')
    results = predict(img_to_test,detection_graph,category_index,args.visualization)

    print('Saving detections')
    #creating destination folder
    os.makedirs(args.output,exist_ok=True)
    format_string='{} {} {} {} {} {}\n'
    for index,val in enumerate(results):
        if len(args.base_folder)>0:
            filename=img_to_test[index].replace(args.base_folder,'')
        else:
            filename = os.path.basename(img_to_test[index])
        filename=filename[:-4]+'.txt'
        classes = val['classes']
        boxes = val['boxes']
        scores = val['scores']
        num_detections = val['num_detections']
        destination = os.path.join(args.output,filename)
        os.makedirs(os.path.abspath(os.path.join(destination, os.pardir)),exist_ok=True)
        with open(destination,'w+') as f_out:
            for idx in range(scores.shape[1]):
                s = scores[0][idx]
                c = classes[0][idx]-1
                ymin, xmin, ymax, xmax = boxes[0][idx]
                x_center = (xmax-xmin)/2+xmin
                y_center = (ymax-ymin)/2+ymin
                w = (xmax-xmin)
                h = (ymax-ymin)
                f_out.write(format_string.format(c,x_center,y_center,w,h,s))
    print('All Done!')
