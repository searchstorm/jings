"""Mask RCNN Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
# wjh
import glob
import os
import time
from gluoncv import model_zoo, data, utils
import numpy as np
import cv2
import re
def parse_args():
    parser = argparse.ArgumentParser(description='Test with Mask RCNN networks.')
    parser.add_argument('--network', type=str, default='mask_rcnn_resnet50_v1b_coco',
                        help="Mask RCNN full network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    # wjh
    parser.add_argument('--imgdir', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Testing with GPUs, you can specify 0 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    # ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()]

    #grab some image if not specified
    # if not args.images.strip():
    #     gcv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
    #                       'gluoncv/detection/biking.jpg?raw=true', 'biking.jpg')
    #     image_list = ['biking.jpg']
    # else:
    #     image_list = [x.strip() for x in args.images.split(',') if x.strip()]


    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False)
        # print(net)
        net.load_parameters(args.pretrained,allow_missing=True,ignore_extra=True)
    net.set_nms(0.3, 200)
    net.collect_params().reset_ctx(ctx)

    # wjh
    if args.imgdir.strip():
        image_list = os.listdir(args.imgdir)
        image_list.sort(key=lambda x:int(x[:-4]))
    # print(image_list)
    filename = 'allserver.txt'

    for image in image_list:
        if args.imgdir.strip():
            x, orig_img = presets.rcnn.load_test(args.imgdir + '/' + image, short=net.short, max_size=net.max_size)
        else:
            x, orig_img = presets.rcnn.load_test(image, short=net.short, max_size=net.max_size)
        x = x.as_in_context(ctx[0])

        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

        print('-----------------')
        #
        #print(scores.shape)
        
        #result_lists = np.zeros((b.shape[0],11))

        width, height = orig_img.shape[1], orig_img.shape[0]
        masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
        results = np.zeros((1000,8))-1
        for k in range(masks.shape[0]):
            y,x = np.where(masks[k]>0)
            x_uniq = np.unique(x)
            points = []

            for i in x_uniq:
                index = np.where(x == i)
                points.append([int(i/800*1024),int(y[index[0][0]]/800*1024)])
                points.append([int(i/800*1024),int(y[index[0][-1]]/800*1024)])
                rect = cv2.minAreaRect(np.array(points))
                results[k] = cv2.boxPoints(rect).reshape(-1)
        # 差框的输出坐标
        #print(scores)
        b = np.round(scores[scores > 0.9],2)
        a = ids[scores > 0.9]
        print(bboxes.shape)

        c = results[(scores > 0.9).reshape(-1)]
        c[c<0] = 0
        if(b.size != 0):

            print(b)
            print(a)
            print(c.shape)
            print(image)
            # 得到的bbox 是左上角和右下角的点坐标(x,y)
            with open(filename, 'a', encoding='utf-8') as f:
                for i in range(b.shape[0]):

                    f.writelines(image + ' ' + str(int(a[i]+1)) + ' '+ str(b[i])+' '+
                                 str(int(c[i][0])) + ' ' + str(int(c[i][1])) + ' '+ str(int(c[i][2])) + ' '+ str(int(c[i][3])) + ' '+
                                 str(int(c[i][4])) + ' ' + str(int(c[i][5])) + ' '+ str(int(c[i][6])) + ' '+ str(int(c[i][7])) + ' '+'\n')


            orig_img = utils.viz.plot_mask(orig_img, masks)
            # identical to Faster RCNN object detection
            plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(1, 1, 1)
            # ax = utils.viz.plot_bbox(orig_img, results, scores, ids,
            #                          class_names=net.classes, ax=ax)
            img = cv2.imread(args.imgdir + '/' + image)
            plt.imshow(img)
            for i in range(c.shape[0]):
                plt.plot([c[i][0],c[i][2],c[i][4],c[i][6],c[i][0]],
                         [c[i][1],c[i][3],c[i][5],c[i][7],c[i][1]],'-b')
            plt.savefig('result/' + image)
            # plt.show()
        '''
        plt.axis('off')
        plt.savefig('./final_sour70/'+image[-12:-4]+'.png')
        '''

        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # 设置保存路径
        # out_png_path = './final_fpn70/'+image[-12:-4]+'.png'

        # 保存图片，并设置保存参数
        # bbox_inches='tight'和pad_inches=0.0都很关键
        # dpi可以调节你保存的图片的清晰度（默认保存的一般清晰度都很感人...）
        # plt.savefig(out_png_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
