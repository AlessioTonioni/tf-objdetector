import argparse
import os

MODEL_NAME_TO_PATH={
    'faster_rcnn_atrous': os.path.join('sample_config','faster_rcnn_inception_resnet_v2_atrous.config'),
    'faster_rcnn_resnet50':os.path.join('sample_config','faster_rcnn_resnet50.config'),
    'faster_rcnn_resnet101':os.path.join('sample_config','faster_rcnn_resnet101.config'),
    'faster_rcnn_resnet152':os.path.join('sample_config','faster_rcnn_resnet152.config'),
    'faster_rcnn_inception_v2':os.path.join('sample_config','faster_rcnn_inception_v2.config'),
    'faster_rcnn_nas':os.path.join('sample_config/','faster_rcnn_nas.config'),
    'rfcn_resnet101':os.path.join('sample_config','rfcn_resnet101.config'),
    'ssd_inception_v2':os.path.join('sample_config','ssd_inception_v2.config'),
    'ssd_mobilenet_v1':os.path.join('sample_config','ssd_mobilenet_v1.config')
}

def check_existance(path):
    if not os.path.exists(path):
        print('ERROR: unable to find {}'.format(path))
        exit()

def count_classes(path_to_pbtxt):
    with open(path_to_pbtxt) as f_in:
        pbtxt = ''.join(f_in.readlines())
    return pbtxt.count('id')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Properly setup the configuration file to be used for the training")
    parser.add_argument('-m','--model',help="name of the model to be used",choices=MODEL_NAME_TO_PATH.keys(),default='ssd_inception_v2')
    parser.add_argument('-t','--training',help='path to the tfrecord with the training set',required=True)
    parser.add_argument('-v','--validation',help='path to the tfrecord to be used for validation',required=True)
    parser.add_argument('-l','--label', help="path to the labelmap file",required=True)
    parser.add_argument('-w','--weights',help="path to the checkpoint path to be used as initialization",required=True)
    parser.add_argument('-o','--output', help="path where the output config file will be saved",default="model.config")
    parser.add_argument('-s','--step',help="max number of training step for the detector",type=int)
    args=parser.parse_args()

    for p in [args.training, args.validation, args.label]:
        check_existance(p)
    
    model_config = MODEL_NAME_TO_PATH[args.model]
    num_classes = count_classes(args.label)

    with open(model_config) as f_in:
        lines = f_in.readlines()

    new_lines=[]
    in_training=False
    in_validation=False
    for l in lines:
        if 'fine_tune_checkpoint' in l:
            l=l.replace('PATH_TO_BE_CONFIGURED',args.weights)
        elif 'num_classes' in l:
            l=l.replace('??',str(num_classes))
        elif 'train_input_reader' in l:
            in_training=True
        elif 'eval_input_reader' in l:
            in_validation=True
        elif 'input_path' in l:
            if in_training:
                l = l.replace('PATH_TO_BE_CONFIGURED',args.training)
                in_training=False
            elif in_validation:
                l = l.replace('PATH_TO_BE_CONFIGURED', args.validation)
                in_validation=False
        elif 'label_map_path' in l:
            l = l.replace('PATH_TO_BE_CONFIGURED', args.label)
        elif 'num_step' in l:
            l = l.replace('??',str(args.step))
        new_lines.append(l)

    with open(args.output,'w+') as f_out:
        f_out.writelines(new_lines)
    
    print('DONE, result config saved to {}'.format(args.output))

