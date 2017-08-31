import os
import argparse
import io
import hashlib
import tensorflow as tf
from PIL import Image


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_tf_record(image_path, label_path, class_names):
    #load image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_img = fid.read()

    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    key = hashlib.sha256(encoded_img).hexdigest()

    width, height = image.size

    #read annotation
    with open(label_path, 'r') as li:
        annotations = li.readlines()

    #convert annotations to tensorflow format
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for a in annotations:
        c_id, x, y, w, h = a.strip().split(' ')
        c_id=int(c_id)
        x=float(x)
        y=float(y)
        w=float(w)
        h=float(h)

        xmin.append(float(x - (w / 2)))
        ymin.append(float(y - (h / 2)))
        xmax.append(float(x + (w / 2)))
        ymax.append(float(y + (h / 2)))
        #class 0 is for background?
        classes.append(c_id+1)
        classes_text.append(class_names[c_id].encode('utf8'))
        #????????????????????????????????????
        truncated.append(0)
        poses.append(''.encode('utf8'))
        difficult_obj.append(int(False))

    #create tfrecords
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(image_path.encode('utf8')),
        'image/source_id': bytes_feature(image_path.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_img),
        'image/format': bytes_feature(image_path[-3:].encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj),
        'image/object/truncated': int64_list_feature(truncated),
        'image/object/view': bytes_list_feature(poses),
    }))

    return example

def convert_yolo_tf(train_file,tf_record_out,label_name_out,class_to_name):
    try:
        file_list=[]
        with open(train_file) as f_in:
            file_list=f_in.readlines()
        
        class_names=[]
        with open(class_to_name) as f_in:
            class_names=f_in.readlines()
        class_names=[c.strip() for c in class_names if len(c)>0]
    except Exception as e:
        print('Caught Exception: {}'.format(e))
        print('Shutting Down')
        exit()
    
    label_list = [f.strip().replace('images','labels')[:-4]+'.txt' for f in file_list]
    

    #create tensorflow writer to write the final tfrecord
    writer = tf.python_io.TFRecordWriter(tf_record_out)

    for idx,(f,l) in enumerate(zip(file_list,label_list)):
        #get a single tfrecord describing the image and annotations
        example = get_tf_record(f.strip(), l, class_names)

        #write tfrecord
        writer.write(example.SerializeToString())

        print('{}/{}'.format(idx, len(file_list)),end='\r')

    print('\nTFRecord saved, creating label_name.pbtxt')
    with open(label_name_out, 'w+') as f_out:
        proto_string="\nitem{{\n\tid: {}\n\tname: '{}' \n }}\n"
        for i,c in enumerate(class_names):
            f_out.write(proto_string.format(i+1,c))
    
    print('Conversion Done')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Converts a training dataset from yolo format to tfrecord to be used with tensorflow object detectors")
    parser.add_argument('-t','--trainingList', help="path to the training/test list file used by yolo",required=True)
    parser.add_argument('-o', '--outputTfRecord', help="where the output tfrecords will be saved", required=True)
    parser.add_argument('-c','--classNameFile',help="path to the file containing class names",required=True)
    parser.add_argument('-l', '--labelMap', help="where the label map file will be saved, leave empty for default", default='')
    args = parser.parse_args()

    result_dir = os.path.abspath(os.path.join(args.outputTfRecord, os.pardir))
    if args.labelMap == '':
        args.labelMap = os.path.join(result_dir,'label_map.pbtxt')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    convert_yolo_tf(args.trainingList, args.outputTfRecord,args.labelMap, args.classNameFile)
