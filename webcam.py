import os
import cv2 
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from niryolib import Robot, sleep
import talk

#Getting our robot ready to scan
myrobot = Robot('/dev/ttyACM0')
sleep(1)
myrobot.move(0, 14000, -10000,0,0)
myrobot.grip_release()

#Object detection start 
print('Runninng script...')
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

user_request = talk.user_request()

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    print(category_index)
    print(detections['detection_classes']+label_id_offset)
    
    # retrieves the corresponding name of the object from a dictionary.
    detected_object_id = detections['detection_classes'][0] + label_id_offset
    detected_object_name = category_index[detected_object_id]['name']
    
    if detected_object_name == user_request and (float(detections['detection_scores'][0]) > 0.7):
        # create category_index_filtered dictionary for user selected objects only
            category_index_filtered = {k: v for k, v in category_index.items() if v['name'] == user_request}
        # visualize boxes and labels only for user selected objects
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index_filtered,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.7,
                agnostic_mode=False)
        
            startX = detections['detection_boxes'][0][1]
            startY = detections['detection_boxes'][0][0]
            endX = detections['detection_boxes'][0][3]
            endY = detections['detection_boxes'][0][2]
            cX = float((startX + endX)/2)
            cY = float((startY + endY)/2)
        
            if cX > 0.58: 
                myrobot.move(-100, 0,0,0,0) 
            if cX < 0.50:
                myrobot.move(100,0,0,0,0)
            if 0.52 < cX < 0.58:
                print(myrobot.distance()) 
                dist = int(myrobot.distance()) * 300 
                myrobot.move(0, dist, -4000, 0 , 0)
                myrobot.grip_catch()
                sleep(1)
                myrobot.move(0,-15000,-1000,0,0)
                break 

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
