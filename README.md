# Utilities to use Tensorflow object detection API (coming from yolo)

This repository provides a bunch of scripts to train and deploy an object detector using tensorflow object detection api.

Follow the step by step guide to train, validate and deploy your own object detector.

**Requires:**

+ Python3
+ tensorflow --> https://www.tensorflow.org/
+ tensorflow object detection api --> https://github.com/tensorflow/models/tree/master/research/object_detection

1. Create a dataset in yolo like format: 

    + 'images' --> folder containing the training image
    + 'labels' --> containing 1 annotation file for each image in .txt (one BB per row with class x-center y-center w h) 
    + 'traininglist.txt' --> a txt file where each row refer an image to be used as training sample, images and labels folder should be contained in the same directory
    + 'validationlist.txt' --> a txt file where each row refer an image to be used as validation sample, images and labels folder should be contained in the same directory
    + 'className.txt' --> a txt file with the name of the class to be displayed, one per row

1. Convert both the training and validation set to tfrecord:

    ``` bash
    python yolo_tf_converter.py \
        -t ${IMAGE_LIST} \
        -o ${OUTPUT} \
        -c ${CLASSES}
    ```
    + IMAGE_LIST: path to traininglist.txt or validationlist.txt
    + OUTPUT: where the output files will be saved (tfrecord+labelmap)
    + CLASSES: path to the className.txt file

1. Create the configuration file for training using the create_config.py script

    ``` bash
    python create_config.py \
        -t ${TRAINING} \
        -v ${VALIDATION} \
        -l ${LABELS} \
        -w ${WEIGHTS} \
        -m ${MODEL} \
        -s ${STEP}
    ```
    + TRAINING,VALIDATION: path to the tfrecord to be used for training and validation
    + LABELS: path to the labelmap.pbtxt
    + WEIGHTS: path to the starting weights of the model, available here --> <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md>
    + MODEL: name of the model to be used
    + STEP: number of training steps for the detector

    If needed change the parameter in the produced 'model.config'

1. Train the model as long as possible:

    ``` bash
    # From the tensorflow/models/research directory
    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --train_dir=${PATH_TO_TRAIN_DIR}
    ```
    + PATH_TO_YOUR_PIPELINE_CONFIG: path to the model.config generated at step 3
    + PATH_TO_TRAIN_DIR: where the model will be saved

1. OPTIONAL - Run evaluation

    ```bash
    # From the tensorflow/models/research directory
    python object_detection/eval.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
        --eval_dir=${PATH_TO_EVAL_DIR}
    ```
    + PATH_TO_EVAL_DIR: where the evaluation event will be saved (use tensorboard to visualize them)

1. Export trained model as inference graph (WARNING: this action freeze the weight, so the model can only be used for inference not for training)

    ``` bash
    # From tensorflow/models/research
    python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
        --output_directory output_inference_graph
    ```

1. Visual Detection 

    ``` bash
    python inference_engine.py \
        -g ${GRAPH} \
        -l ${LABEL_MAP} \
        -t ${TARGET} \
        -o ${OUT_FLD}
        -v
    ```

    + GRAPH: path to the frozen inference graph produced at step 6
    + LABEL_MAP: path to labelmap.pbtxt
    + TARGET: path to an image to test or to a .txt file with the list of image to test (one per row)
    + OUT_FLD: folder were the prediction will be saved, one '.txt' file for each image with one detection per row encoded as: %class %X_center %Y_center %width %height %confidence
    
1. OPTIONAL - Live detection from webcam (needs opencv...)
    
    ``` bash
    python webcam_detection.py \
        -g ${GRAPH} \
        -l ${LABEL_MAP} \
        -c ${CAM_ID}
    ```
    
    + GRAPH: path to the frozen inference graph produced at step 6
    + LABEL_MAP: path to labelmap.pbtxt
    + CAM_ID: id of the camera as seen by opencv
