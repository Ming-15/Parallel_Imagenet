# _*_ coding:utf-8 _*_
'''
@author: Ming
@time: 12.02.2021
@target: Data Parallel
      __      __
     &&&     &&&              &&
    && $$   $$ &&            &&
   &&   $$ $$   &&          &&
  &&     $$$     &&        &&
 &&      £¤       &&      &&
&&                 &&    &&&&&&&&&&&&
'''

# coding=utf-8
import os
# import args
# import train_func
import help_api
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime as dt
import numpy as np
from torchvision.models import alexnet
import NN_train_help as nth
import pickle

torch.cuda.set_device(0)

torch.manual_seed(15)  # 1515
model_save_path = './feature_model_path'


# pre args
class pre_args(object):
    def __init__(self):
        # self.batch_size = 60
        self.batch_size = 2048
        self.workers = 16
        # self.epoches = 350
        # self.epoch_start = 0
        # self.epoch_end = 50
        self.num_epochs = 50
        self.l_rate = 0.01
        self.lr_update_step = 50
        self.lr_decay = 0.5


args = pre_args()

# load model
# 2 GPUs
args.gpu_id = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device_ids = range(torch.cuda.device_count())

use_a_best = True
canum = 1000
model = alexnet(pretrained=False, num_classes=canum)

model_base_name = 'alex_init25_V1_' + str(canum)

feature_model_save_path = help_api.make_dir(model_save_path, model_base_name)
save_path_for_transfer = os.path.join(feature_model_save_path, 'transfer_records.pkl')
# load data
if True:
    # imagenet = '/data3/AI/imagenet'+str(canum)+'_split'
    imagenet = '/data3/AI/projects/p2022/data/imagenet'
    tr_imagenet = os.path.join(imagenet, 'train')
    te_imagenet = os.path.join(imagenet, 'test')
    # imagenet = '/data3/AI/projects/p2022/data/imagenet/val'
    normalize_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize_te = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr_load_data = datasets.ImageFolder(tr_imagenet, transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize_tr,
    ]))
    te_load_data = datasets.ImageFolder(te_imagenet, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_te,
    ]))
    # all_data = len(tr_load_data)
    # trl = int(all_data*0.8)
    # tr,te = torch.utils.data.random_split(tr_load_data,[trl,all_data-trl])
    label_class = tr_load_data.class_to_idx
    # print(tr_load_data.class_to_idx)
    print('the number of cats:', len(tr_load_data.classes))
    train_data = torch.utils.data.DataLoader(tr_load_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
    val_data = torch.utils.data.DataLoader(te_load_data, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers, pin_memory=True)



if not use_a_best:
    model_file_path = os.path.join(feature_model_save_path, 'alexnetit.pkl')
    total_epochs = 0
    save_model_dict = {}
    val_list = []
    lr_list = []
    avg_train_losses = []
    avg_valid_losses = []
    tr_acc_list = []
    transfer_count = 0
else:
    model_file_path = os.path.join(feature_model_save_path, 'a_best.pkl')
    args.l_rate = 0.008
    args.lr_decay = 0.75
    args.num_epochs = 25
    args.lr_update_step = 12
    with open(save_path_for_transfer, 'rb') as f:
        temp = pickle.load(f)
        save_model_dict = temp['save_model_dict']
        best_acc = temp['best_acc']
        total_epochs = temp['total_epochs']
        val_list = temp['val_list']
        lr_list = temp['lr_list']
        avg_train_losses = temp['avg_train_losses']
        avg_valid_losses = temp['avg_valid_losses']
        transfer_count = temp['transfer_count']
        tr_acc_list = temp['tr_acc_list']
    print('-------------------pretrained-------------------')


print(model_file_path)
model.load_state_dict(torch.load(model_file_path, map_location='cpu'))
model.cuda()
model = nn.DataParallel(model, device_ids=[0, 1])

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.9)
# optimizer = torch.optim.AdamW(model.parameters(),lr=args.l_rate,weight_decay=args.lr_decay)
# model.optimizer = torch.optim.AdamW(model.parameters(),lr=args.l_rate)
# # model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_step, gamma=args.lr_decay)


if not use_a_best:
    accuracy = nth.easy_validate(model, val_data)
    print('original accuracy:', accuracy)
    torch.save(model.module.state_dict(), os.path.join(feature_model_save_path, str(accuracy) + '_acc_001.pkl'))
    best_acc = accuracy

save_model_id = [i for i in range(101)]


# init training
t_start = dt.datetime.now()
print(t_start)
batch_num = len(train_data)
bs = train_data.batch_size
print('batch-size:', bs, '  batch-num:', batch_num)
frequency = 5
show_steps = [int(batch_num * frequency / 100) * i for i in range(int(100 / frequency))]

for num_epoch in range(args.num_epochs):
    print('epoch:', str(num_epoch), '----------------------')
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []

    cell_num = 0
    correct = 0  # cal the accuracy
    ###################
    # train the model #
    ###################
    model.train()
    for c, (data, label) in enumerate(train_data):
        if c in show_steps:
            it = int(c/batch_num*100)
            nth.visual_progross(it)
        data = data.cuda()
        label = label.cuda()
        out = model.forward(data)
        _, predicted = torch.max(out.data, 1)
        cell_num += label.size(0)
        correct += (predicted == label.data).sum()
        # ou = model.forward(cell)
        loss = criterion(out, label)
        print_loss = loss.data.item()
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # if num_epoch<50:
    #     scheduler.step()
    #     scheduler.step()
    # else:
    scheduler.step()
    tr_acc = round(correct.item() / cell_num, 3)
    averaged_batch_loss = np.average(train_losses)
    print('**********train_acc:', tr_acc, 'averaged batch loss', averaged_batch_loss)

    tr_acc_list.append(tr_acc)
    t_ep = dt.datetime.now()
    print('***********epoch' + str(num_epoch) + 'consume time:', t_ep - t_start)
    ######################
    # validate the model #
    ######################
    accuracy = nth.validate(model, val_data, criterion, valid_losses)
    val_list.append(accuracy)
    # save models
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.module.state_dict(), os.path.join(feature_model_save_path, 'a_best.pkl'))
    ac_id = str(accuracy)[2:4]
    if int(ac_id) in save_model_id:
        temp_model_path = os.path.join(feature_model_save_path, 'acc_' + ac_id + '.pkl')
        if not os.path.exists(temp_model_path):
            print('saving model:', ac_id)
            torch.save(model.module.state_dict(), temp_model_path)
            save_model_dict[ac_id] = accuracy
        else:
            if accuracy > save_model_dict[ac_id]:
                torch.save(model.module.state_dict(), temp_model_path)
                save_model_dict[ac_id] = accuracy
    if True:
        # update
        lr_list.append(scheduler.get_last_lr()[0])
        avg_train_losses.append(averaged_batch_loss)
        avg_valid_losses.append(np.average(valid_losses))
        epoch_len = len(str(args.num_epochs))
        print_msg = (f'[{num_epoch:>{epoch_len}}/{args.num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {avg_train_losses[-1]:.5f} ' +
                     f'valid_loss: {avg_valid_losses[-1]:.5f}')
        print(print_msg, '/n', 'validation accuracies', val_list)
    total_epochs += 1
    result = {'transfer_count': transfer_count,
              # use append
              'val_list': val_list, 'lr_list': lr_list, 'avg_train_losses': avg_train_losses,
              'avg_valid_losses': avg_valid_losses, 'tr_acc_list': tr_acc_list,
              # use update
              'save_model_dict': save_model_dict, 'best_acc': best_acc, 'total_epochs': total_epochs}
    with open(save_path_for_transfer, 'wb') as f:
        pickle.dump(result, f)
        f.close()

transfer_count += 1
# #'val accuracy':val_list,'learning rate':lr_list,'train loss':avg_train_losses,'val loss':avg_valid_losses,
result = {'transfer_count': transfer_count,
          # use append
          'val_list': val_list, 'lr_list': lr_list, 'avg_train_losses': avg_train_losses,
          'avg_valid_losses': avg_valid_losses,'tr_acc_list': tr_acc_list,
          # use update
          'save_model_dict': save_model_dict, 'best_acc': best_acc, 'total_epochs': total_epochs}

with open(save_path_for_transfer, 'wb') as f:
    pickle.dump(result, f)
print(result)

# import pickle
# ap = os.path.join('/data3/AI/projects/p2022/feature_model_path/alex_init25_1000','transfer_records.pkl')
# with open(ap, 'wb') as f:
#     pickle.dump(temp,f)
#     # temp = pickle.load(f)



# transfer_count = 1
# val_list = temp['val accuracy']
# lr_list = temp['learning rate']
# avg_train_losses = temp['train loss']
# avg_valid_losses = temp['val loss']
# tr_acc_list = temp['val accuracy']
# save_model_dict = {}
# for accuracy in val_list:
#     ac_id = str(accuracy)[2:4]
#     temp_model_path = os.path.join('/data3/AI/projects/p2022/feature_model_path/alex_init25_1000', 'acc_' + ac_id + '.pkl')
#     if os.path.exists(temp_model_path):
#         if ac_id not in save_model_dict.keys():
#             save_model_dict[ac_id] = accuracy
#         else:
#             if accuracy > save_model_dict[ac_id]:
#                 save_model_dict[ac_id] = accuracy
# best_acc = max(val_list)
# total_epochs = 45
# result = {'transfer_count': transfer_count,
#           # use append
#           'val_list': val_list, 'lr_list': lr_list, 'avg_train_losses': avg_train_losses,
#           'avg_valid_losses': avg_valid_losses,'tr_acc_list': tr_acc_list,
#           # use update
#           'save_model_dict': save_model_dict, 'best_acc': best_acc, 'total_epochs': total_epochs}


# # training with new arg
#
# for num_epoch in range(1,1+args.epoches):
#     print('epoch:', str(num_epoch), '----------------------')
#     model, loss = train_func.train_model(train_data=train_loader, epoch=num_epoch, model=model, model_tag=1)
#     loss_list.append(loss)
#     print('loss',loss)
#     help_api.draw_multiple_line(y_list=[loss_list],name=['loss'],stroe_path=os.path.join(args.figure_path,str(num_epoch)+'.png'))
#     t_ep = dt.datetime.now()
#     print('epoch' + str(num_epoch) + 'consume time:')
#     print(t_ep - t_start)
#     if num_epoch > 1:
#         tm = max(te_acc_list)
#     te_ac = train_func.validate(model, te_loader)
#     te_acc_list.append(te_ac)
#     print(te_ac,'test_accuracy')
#     if te_ac >= 0.4:
#         if num_epoch > 2:
#             ta = int(te_ac * 100)
#             model_path = os.path.join(args.model_path, str(ta))
#             if not os.path.exists(model_path):
#                 os.mkdir(model_path)
#                 to_store.append(ta)
#                 torch.save(model.state_dict(), model_path + '/' + str(ta) + '.pkl')
#             else:
#                 pass
#
#
# help_api.draw_multiple_line([ te_acc_list], [ 'test'],stroe_path=os.path.join(args.figure_path,'accuracy.png'))
# # help_api.draw_multiple_line([val_acc_list, te_acc_list], ['validate', 'test'],stroe_path=os.path.join(figure_path,'accuracy.png'))

