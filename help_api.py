# _*_ coding:utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt
import os
# import cv2
# import torch
def write_ind(alist,name):
    fileObject = open(name+'.txt', 'w')
    for ip in alist:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
def make_dir(pa, d):
    patht = os.path.join(pa, d)
    if not os.path.exists(patht):
        os.mkdir(patht)
    return patht

def compute_confusion_matrix(pre,label):
    _,pre_l = np.unique(pre,return_inverse=True)
    names,label_l = np.unique(label,return_inverse=True)
    l = len(names)
    cm = np.zeros((l,l))
    cm = list(cm)
    cm = [list(x) for x in cm]
    for i in range(len(label_l)):
        cm[label_l[i]][pre_l[i]] += 1
    for i in range(len(cm)):
        t=sum(cm[i])
        cm[i].append(t)
    return cm

def draw_line(y_list,x_list = None):
    if x_list==None:
        l = len(y_list)
        x_list = [i for i in range(1,l+1)]
    plt.plot(x_list,y_list)
    plt.title('test_acc')
    plt.xlabel('num_epoch')
    plt.ylabel('class_acc')
    plt.show()
def jp_draw_line(y_list,x_list = None):
    if x_list==None:
        l = len(y_list)
        x_list = [i for i in range(1,l+1)]
    plt.plot(x_list,y_list)
    plt.title('test_acc')
    plt.xlabel('num_epoch')
    plt.ylabel('class_acc')
    plt.show()
def draw_multiple_line(y_list,name=None,stroe_path = None):
    plt.figure()
    num = len(y_list)
    tl = []
    for i in range(num):
        x_list = [i for i in range(1, len(y_list[i]) + 1)]
        ftl, = plt.plot(x_list, y_list[i])
        tl.append(ftl)
    if name != None:
        plt.legend(handles = tl,labels = name,loc='best')
    plt.title('performance')
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')
    if stroe_path is None:
        plt.show()
        import time
        time.sleep(10)
        plt.close()
    else:
        plt.savefig(stroe_path)
        plt.close()
def jp_draw_multiple_line(y_list,name=None,stroe_path = None):
    plt.figure()
    num = len(y_list)
    tl = []
    for i in range(num):
        x_list = [i for i in range(1, len(y_list[i]) + 1)]
        ftl, = plt.plot(x_list, y_list[i])
        tl.append(ftl)
    if name != None:
        plt.legend(handles = tl,labels = name,loc='best')
    plt.title('performance')
    plt.xlabel('epoch_num')
    plt.ylabel('accuracy')
    if stroe_path is None:
        plt.show()
        import time
        time.sleep(10)
        plt.close()
    else:
        # for jupyter notebook do not always need to show
        plt.savefig(stroe_path,dpi=300,face_color='white')
        plt.show()
        plt.close()

if __name__ == '__main__':
    y_true = [1,2,0,3,4,1,2,4,1,2,1,3,1,1]
    y_pred = [1,2,0,3,4,1,2,4,2,1,3,1,1,4]
    x = [i for i in range(20)]
    x1 = [i*i for i in range(20)]
    x2 = [i**3 for i in range(20)]
    draw_multiple_line([x,x1,x2],['x1','x2','x3'])
    # x = [i for i in range(14)]
    # draw_line([[1,2],[2,3],[1,2],[2,3],[1,2],[2,3],[1,2]])
    # # c = compute_confusion_matrix(y_pred,y_true)