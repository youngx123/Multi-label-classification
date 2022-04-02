# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:00  2022-02-27
import os
import time
from darknet19 import DarkNet
from dataloder import ImageLoader
from torch.utils.data import DataLoader
import dataloder
import torch
import numpy as np
import cv2
from darknet53 import darknet53

def main(savedir):
    device = "cuda"
    class_num = len(dataloder.TYPE)+len(dataloder.COLOR)
    # net = DarkNet(class_num)
    net = DarkNet(class_num=class_num)
    net.load_state_dict(torch.load("best.pt"))
    valdir = "./test_data"
    valset = ImageLoader(valdir)
    valLoader = DataLoader(valset, batch_size=1, shuffle=False)

    net.to(device)
    net.eval()
    pred_num = 0
    with torch.no_grad():
        for id, batch in enumerate(valLoader):
            img, label, imgfile = batch[0].float().to(device), batch[1].float().to(device), batch[2]
            label = label.squeeze(1)
            color_gt = label[0, :len(dataloder.COLOR)]
            type_gt = label[0, len(dataloder.COLOR):]

            _, color_gt_index = torch.max(color_gt, 0)
            _, type_gt_index = torch.max(type_gt, 0)

            imagename = "{}_{}_{}.png".format(dataloder.COLOR[int(color_gt_index)], dataloder.TYPE[int(type_gt_index)], str(id))

            startTime = time.time()
            pred = net(img)
            image = cv2.imread(imgfile[0])

            color_pred = pred[0, :len(dataloder.COLOR)]
            type_pred = pred[0, len(dataloder.COLOR):]
            color_max_prob, color_cls_index = torch.max(color_pred, 0)
            type_max_prob, type_cls_index = torch.max(type_pred, 0)
            if color_cls_index.item()==color_gt_index and type_cls_index.item() ==type_gt_index:
                pred_num +=1
            color_mess = '%s : %.3f' % (dataloder.COLOR[int(color_cls_index)], color_max_prob)
            cv2.putText(image, color_mess, (int(10), int(200 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

            type_mess = '%s : %.3f' % (dataloder.TYPE[int(type_cls_index)], type_max_prob)
            cv2.putText(image, type_mess, (int(10), int(300 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)

            cv2.imwrite(savedir + "/" +imagename, image)
            endTime = time.time()
            print("%s use time %.3f"%(os.path.basename(imgfile[0]), endTime - startTime))

    print("eval acc : %.3f , total num %d,  %d" % (pred_num / len(valLoader), pred_num, len(valLoader)))


if __name__ == '__main__':
    saveDir = "./test_result/"
    os.makedirs(saveDir, exist_ok=True)
    main(saveDir)
