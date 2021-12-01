import argparse
import logging
import time
import datetime

import cv2
import numpy as np
import pandas as pd
import os

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.python_cv2 import VideoCapture
from darts_convert import *

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def parser():
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()
    return args

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def dir_check(ta):
#    if not os.path.exists('./csv/{}'.format(ta)):
#        os.mkdir('./csv/{}'.format(ta))
    if not os.path.exists('./video/{}'.format(ta)):
        os.mkdir('./video/{}'.format(ta))
#    if not os.path.exists('./image/{}'.format(ta)):
#        os.mkdir('./image/{}'.format(ta))
    logger.debug('dir_check ok+')
    
def dart_cam(now,args,videoname):
    fps_time = 0
    ta = now.strftime('%Y_%m%d')
    dir_check(ta)

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = VideoCapture(args.camera)
    image = cam.read()

    width = 1280;height = 720;fps = 7

    dfs = pd.DataFrame(index=[])
    columns = ['frame','human', 'point', 'x', 'y']
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(videoname, fourcc, fps, (width, height))
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    hu = []
    i = 0
    
    while True:
        image = cam.read()
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)        
        image_h, image_w = image.shape[:2]
        xx = 0
        for human in humans:
            xx = xx + 1
            for m in human.body_parts: 
                body_part = human.body_parts[m]
                #print(body_part)
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                list = [[i,xx, m, center[0],center[1]]]
                df = pd.DataFrame(data=list, columns=columns)
                dfs = pd.concat([dfs, df])
        i += 1
        logger.debug('show+')
        cv2.putText(image,"FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        writer.write(image)
        fps_time = time.time()
        if i==30:
            break
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
    writer.release()

    return dfs

def main():
    
    logger = logging.getLogger('TfPoseEstimator-WebCam')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    now = datetime.datetime.now()
    dir_name = now.strftime('%Y_%m%d')
    dir_check(dir_name)
    args = parser()
    videoname = './video/{}'.format(now.strftime('%Y_%m%d'))+ '/' + now.strftime('%Y%m%d_%H%M') + '.mp4'

    body_data = dart_cam(now,args,videoname)
    frame,rad = convert_rad(body_data)
    test_list = rad_convert_nor(rad)
    train_list = [1.15E+02,1.12E+02,1.10E+02,9.75E+01,7.93E+01,6.69E+01,6.03E+01,1.57E+02,1.74E+02,1.70E+02]
    train_list = rad_convert_nor(train_list)

    path, cost = partial_dtw(test_list,train_list)
    print('your score is ',cost)
    D = (np.array(test_list).reshape(1, -1) - np.array(train_list).reshape(-1, 1))**2
    [plt.plot(line, [test_list[line[0]], train_list[line[1]]], linewidth=0.8, c='gray') for line in path]
    plt.plot(test_list)
    plt.plot(train_list)
    plt.plot(path[:,0],test_list[path[:,0]], c='C2')
    plt.show()
        
if __name__ == '__main__':
    main()
