import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

from distutils.version import StrictVersion
import cv2

sys.path.append("..")
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your tensorflow installation to v 1.9.* or later')

from object_detection.utils import label_map_util
from utils import draw_boxes_and_labels

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_FROZEN_MODEL_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                        feed_dict={image_tensor: image_np_expanded})

    rect_points, class_names, class_colors = draw_boxes_and_labels(boxes=np.squeeze(boxes),
                                                                   classes=np.squeeze(classes).astype(np.int32),
                                                                   scores=np.squeeze(scores),
                                                                   category_index=category_index,
                                                                   min_score_thresh=.5)
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


if __name__ == '__main__':
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_MODEL_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    frame = cv2.imread('./test_images/image2.jpg')
    height, width, channels = frame.shape
    input_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data = detect_objects(input_img_rgb, sess, detection_graph)

    font = cv2.FONT_HERSHEY_SIMPLEX
    rec_points = data['rect_points']
    class_names = data['class_names']
    class_colors = data['class_colors']
    for point, name, color in zip(rec_points, class_names, class_colors):
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                      (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                      (int(point['xmin'] * width) + len(name[0]) * 6,
                       int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
        cv2.putText(frame, name[0], (int(point['xmin'] * width), int(point['ymin'] * height)), font,
                    0.3, (0, 0, 0), 1)

    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
