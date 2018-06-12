#encoding=utf8
import sys
import argparse
# sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

_ckpt_path = 'models/pretrain/yolo_tiny.ckpt'

_threshold = 0.1
_iou_threshold = 0.5

def process_predicts(predicts):
  print predicts.shape # (1, 7, 7, 30)
  p_classes = predicts[0, :, :, 0:20]
  C = predicts[0, :, :, 20:22]
  coordinate = predicts[0, :, :, 22:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes
  print P.shape # (7, 7, 2, 20)
  index = np.argmax(P)
  print index # 486
  index = np.unravel_index(index, P.shape)
  print index # (1, 5, 0, 6)

  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0
  if xmin < 0:
    xmin = 0
  if ymin < 0:
    ymin = 0
  xmax = xmin + w
  ymax = ymin + h
  if xmax > 447:
    xmax = 447
  if ymax > 447:
    ymax = 447
  return xmin, ymin, xmax, ymax, class_num


def process_predicts_all(predicts, threshold=_threshold):
  print predicts.shape # (1, 7, 7, 30)
  p_classes = predicts[0, :, :, 0:20]
  C = predicts[0, :, :, 20:22]
  boxes = predicts[0, :, :, 22:]
  boxes = np.reshape(boxes, (7, 7, 2, 4))
  offset_x = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))
  offset_y = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(2,1,0))
  boxes[:,:,:,0] += offset_x
  boxes[:,:,:,1] += offset_y
  boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
  # boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
  # boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  # P = C * p_classes
  P = np.zeros((7,7,2,20))
  for i in range(2):
    for j in range(20):
      P[:,:,i,j] = np.multiply(C[:,:,i,0], p_classes[:,:,0,j])
  print P.shape # (7, 7, 2, 20)
  # index = np.argmax(P)
  filter_mat_probs = np.array(P>=threshold, dtype='bool')
  filter_mat_boxes = np.nonzero(filter_mat_probs)
  print filter_mat_probs.shape
  # print filter_mat_probs
  print filter_mat_boxes

  boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
  probs_filtered = P[filter_mat_probs]
  classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
  print boxes_filtered
  print probs_filtered
  print classes_num_filtered

  # 去除iou大于阈值的部分
  for i in range(len(boxes_filtered)):
    if probs_filtered[i] == 0:
      continue
    for j in range(i+1,len(boxes_filtered)):
			if iou(boxes_filtered[i],boxes_filtered[j]) > _iou_threshold : 
				probs_filtered[j] = 0.0
  filter_iou = np.array(probs_filtered>0.0,dtype='bool')
  boxes_filtered = boxes_filtered[filter_iou]
  probs_filtered = probs_filtered[filter_iou]
  classes_num_filtered = classes_num_filtered[filter_iou]
  return boxes_filtered, classes_num_filtered, probs_filtered

def iou(box1,box2):
  tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
  lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
  if tb < 0 or lr < 0 :intersection = 0
  else : intersection =  tb*lr
  return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default='cat.jpg')
  parser.add_argument('--output', default='cat_out.jpg')
  parser.add_argument('--ckpt', default='')
  return parser.parse_args()

def main(args):
  common_params = {'image_size': 448, 'num_classes': 20, 
                  'batch_size':1}
  net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

  net = YoloTinyNet(common_params, net_params, test=True)

  image = tf.placeholder(tf.float32, (1, 448, 448, 3))
  predicts = net.inference(image)

  sess = tf.Session()

  img = cv2.imread(args.input)
  img_h, img_w = img.shape[:2]
  resized_img = cv2.resize(img, (448, 448))
  np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
  scale_w = 448.0 / img_w
  scale_h = 448.0 / img_h

  np_img = np_img.astype(np.float32)

  np_img = np_img / 255.0 * 2 - 1
  np_img = np.reshape(np_img, (1, 448, 448, 3))

  saver = tf.train.Saver(net.trainable_collection)

  saver.restore(sess, _ckpt_path)

  np_predict = sess.run(predicts, feed_dict={image: np_img})

  # xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
  boxes, classes, _ = process_predicts_all(np_predict)
  for i, box in enumerate(boxes):
    class_num = classes[i]
    class_name = classes_name[class_num]
    x, y, w, h = box
    xmin, ymin, xmax, ymax = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    # xmin, xmax, ymin, ymax = box
    # print xmin, ymin, xmax, ymax, class_num, class_name
    # xmin /= scale_w
    # xmax /= scale_w
    # ymin /= scale_h
    # ymax /= scale_h
    xmin *= img_w
    ymin *= img_h
    xmax *= img_w
    ymax *= img_h
    print xmin, ymin, xmax, ymax, class_num, class_name 
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 4)
    cv2.putText(img, class_name, (int(xmin), int(ymin)), 4, 1.5, (0, 0, 255))
  cv2.imwrite(args.output, img)
  sess.close()


if __name__ == '__main__':
  main(get_args())