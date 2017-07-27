# Utilities to use Tensorflow object detection API (coming from yolo)

This repository provides a bunch of scripts to train and deploy an object detector using tensorflow object detection api.

Follow the step by step guide to train, validate and deploy your own object detector 

**Requires:**
+ Python3
+ tensorflow --> https://www.tensorflow.org/
+ tensorflow object detection api --> https://github.com/tensorflow/models/tree/master/object_detection

1. Create a dataset in yolo like format: 

    + 'images' --> folder containing the training image
    + 'labels' --> containing 1 annotation file for each image in .txt (one BB per row with class x-center y-center w h) 
    + 'traininglist.txt' --> a txt file where each row refer an image to be used as trainign sample, images and labels folder should be contained in the same directory
    + 'className.txt' --> a txt file with the name of the class to be displayed, one per row

2. Convert the dataset to tfrecord:

    ``` bash
    python yolo_tf_converter.py \
        -t ${TRAINING_LIST} \
        -o ${OUTPUT} \
        -c ${CLASSES}
    ```
    + TRAINING_LIST: path to traininglist.txt
    + OUTPUT: where the trainign set will be saved (tfrecord+labelmap)
    + CLASSES: path to the className.txt file

3. Create the configuration file for training using the create_config.py script

    ``` bash
    python create_config.py \
        -t ${TRAINING} \
        -v ${VALIDATION} \
        -l ${LABELS} \
        -w ${WEIGHTS} \
        -m ${MODEL}
    ```
    + TRAINING,VALIDATION: path to the tfrecord to be used for trainign and validation
    + LABELS: path to the labelmap.pbtxt
    + WEIGHTS: path to the starting weights of the model, availabel here --> <https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md>
    + MODEL: name of the model to be used

    If needed change the parameter in the produced 'model.config'

4. Train the model as long as possible:

    ``` bash
    # From the tensorflow/models/ directory
    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --train_dir=${PATH_TO_TRAIN_DIR}
    ```
    + PATH_TO_YOUR_PIPELINE_CONFIG: path to the model.config generated at step 3
    + PATH_TO_TRAIN_DIR: where the model will be saved

5. Run evaluation

    ```bash
    # From the tensorflow/models/ directory
    python object_detection/eval.py \
        --logtostderr \
        --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
        --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
        --eval_dir=${PATH_TO_EVAL_DIR}
    ```
    + PATH_TO_EVAL_DIR: where the evaluation event will be saved (use tensorboard to visualize them)

6. Export trained model as inference graph (WARNING: this action freeze the weight, so the model can only be used for inference not for training)

    ``` bash
    # From tensorflow/models
    python object_detection/export_inference_graph \
        --input_type image_tensor \
        --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
        --checkpoint_path model.ckpt-${CHECKPOINT_NUMBER} \
        --inference_graph_path output_inference_graph.pb
    ```

7. Visual Detection 

    ``` bash
    python inference_engine.py \
        -g ${GRAPH} \
        -l ${LABEL_MAP} \
        -t ${TARGET} \
        -v
    ```

    + GRAPH: path to the inference graph produced at step 6
    + LABEL_MAP: path to labelmap.pbtxt
    + TARGET: path to an image to test or to a .txt file with the list of image to test (one per row)
