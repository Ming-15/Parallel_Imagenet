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
# _*_ coding:utf-8 _*_
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
torch.cuda.set_device(0)

class alexnet_with_norm(nn.Module):
    def __init__(self, num_classes=1000, use_pretrain=True,means=None,stds=None):
        super(alexnet_with_norm, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x

torch.manual_seed(15)#1515
model_save_path = './feature_model_path'
#pre args
class pre_args(object):
    def __init__(self):
        # self.batch_size = 60
        self.batch_size = 2048
        self.workers = 16
        self.epoches = 350
        self.l_rate = 0.01
        self.lr_update_step = 80
        self.lr_decay = 0.1

args = pre_args()

#load model
args.gpu_id="0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device_ids=range(torch.cuda.device_count())
# model =  alexnet_with_norm(canum)
canum=100
model = alexnet(pretrained = False,num_classes=canum)
# feature_model = models.alexnet()
model_file_name = 'AlexNet'+str(canum)+'.pkl'
model_name = model_file_name[:-4]
feature_model_save_path = help_api.make_dir(model_save_path,model_name)
# model.load_state_dict(torch.load(os.path.join(feature_model_save_path,'alexnet25.pkl'),map_location='cpu'))
# print(os.path.join(feature_model_save_path,'alexnet25.pkl'))
model.cuda()
model = nn.DataParallel(model, device_ids=[0,1])
# model.load_state_dict(torch.load(os.path.join(feature_model_save_path,  'a_best.pkl'),map_location='cpu'))

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate,momentum=0.8)
# optimizer = torch.optim.AdamW(model.parameters(),lr=args.l_rate,weight_decay=args.lr_decay)
# model.optimizer = torch.optim.AdamW(model.parameters(),lr=args.l_rate)
# # model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_step, gamma=args.lr_decay)

imagenet = os.path.join('data','imagenet_'+str(canum)+'categories') # data_path
# imagenet = os.path.join(data_path,'imagenet')
tr_imagenet = os.path.join(imagenet,'train')
te_imagenet = os.path.join(imagenet,'test')

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
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize_te,
    ]))
# all_data = len(tr_load_data)
# trl = int(all_data*0.8)
# tr,te = torch.utils.data.random_split(tr_load_data,[trl,all_data-trl])
label_class = tr_load_data.class_to_idx
print(tr_load_data.class_to_idx)
print(len(tr_load_data.classes))
train_data = torch.utils.data.DataLoader(tr_load_data, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True)
val_data = torch.utils.data.DataLoader(te_load_data, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)
#finish load data

val_acc_list = []
te_acc_list = []

val_list = []
tr_acc_list = []
to_store = []
tm = -1
loss_list = []
lr_list = []
save_model_id = [i for i in range(101)]

with torch.no_grad():
    # test the change level
    model.eval()  # prep model for evaluation
    # val_ac
    test_total = 0
    correct = 0
    # eval_number = 0
    for cell, cl in val_data:
        cell = cell.cuda()
        cl = cl.cuda()
        outputs = model.forward(cell)
        # print(cell.data.shape)
        # print(outputs.data.shape)
        _, predicted = torch.max(outputs.data, 1)
        test_total += cl.size(0)
        correct += (predicted == cl.data).sum()
    k = correct.item()
    accuracy = k / test_total
    torch.save(model.module.state_dict(), os.path.join(feature_model_save_path,   str(accuracy)+'_acc_001.pkl'))
    print('original accuracy:',accuracy)
    best_acc = accuracy # init best_acc with change level

t_start = dt.datetime.now()
print(t_start)
batch_num = len(train_data)
bs = train_data.batch_size
print('batch-size:', bs, '  batch-num:', batch_num)
n_epochs = args.epoches
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
# model, loss, tr_acc = train_func.train_model(train_data=tr_data, epoch=num_epoch+1, model=model,model_tag=1)
for num_epoch in range(0, n_epochs):
    print('epoch:', str(num_epoch), '----------------------')
    cell_num = 0
    correct = 0  # cal the accuracy
    mpt1 = []  # store loss
    c = 0
    k = [i for i in range(0, 100, 5)]
    ###################
    # train the model #
    ###################
    model.train()
    for data, label in train_data:
        c += 1
        it = int(c / batch_num * 100)
        if it in k:
            k.remove(it)
            lft = '-' * it
            lrt = ' ' * (100 - it)
            lt = lft + lrt
            print('epoch-execution' + str(it).zfill(2) + '%:' + lt)
        data = data.cuda()
        label = label.cuda()
        out = model.forward(data)
        _, predicted = torch.max(out.data, 1)
        cell_num += label.size(0)
        correct += (predicted == label.data).sum()
        loss = criterion(out, label)
        print_loss = loss.data.item()
        train_losses.append(loss.item())
        print_loss = float(print_loss)
        mpt1.append(print_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # if num_epoch<50:
    #     scheduler.step()
    #     scheduler.step()
    # else:
    scheduler.step() # update epoch para
    loss = sum(mpt1)
    tr_acc = round(correct.item() / cell_num, 3)
    print('**********train_acc:', tr_acc, '    batch loss',loss/batch_num)
    # return model, loss, tr_acc
    loss_list.append(loss)
    tr_acc_list.append(tr_acc)
    t_ep = dt.datetime.now()
    print('***********epoch' + str(num_epoch) + 'consume time:',t_ep - t_start)
    ######################
    # validate the model #
    ######################
    with torch.no_grad():
        model.eval()  # prep model for evaluation
        # val_ac
        test_total = 0
        correct = 0
        # eval_number = 0
        for cell, cl in val_data:
            cell = cell.cuda()
            cl = cl.cuda()
            outputs = model.forward(cell)
            loss = criterion(outputs, cl)
            valid_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            test_total += cl.size(0)
            correct += (predicted == cl.data).sum()
            # eval_number += correct.item()
        k = correct.item()
        # print("correct_item", correct.item())
        # print("eval_number",eval_number)
        accuracy = k / test_total
        val_list.append(accuracy)
        print(accuracy, 'validation_accuracy')
        if accuracy>best_acc:
            torch.save(model.module.state_dict(),os.path.join(feature_model_save_path,  'a_best.pkl'))
        ac_id = str(accuracy)[2:4]
        if int(ac_id) in save_model_id: # save the model with specific accuracy your want
            temp_model_path =os.path.join(feature_model_save_path, 'acc_'+ ac_id + '.pkl')
            if not os.path.exists(temp_model_path):
                print('saving model:',ac_id)
                torch.save(model.module.state_dict(),temp_model_path)
    # update
    lr_list.append(scheduler.get_last_lr()[0])
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    epoch_len = len(str(n_epochs))
    print_msg = (f'[{num_epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')
    print(print_msg)
    # clear lists to track next epoch
    print('validation accuracy',val_list)
    train_losses = []
    valid_losses = []

result = {'val accuracy':val_list,'val learn rate':lr_list,'train loss':avg_train_losses,'val loss':avg_valid_losses}
import pickle
comp_save_path =os.path.join(feature_model_save_path,  'accuracys.pkl')
with open(comp_save_path, 'wb') as f:
    pickle.dump(result, f)
print(result)



help_api.draw_multiple_line([ te_acc_list], [ 'test'],stroe_path=os.path.join(model_save_path,'accuracy.png'))
# help_api.draw_multiple_line([val_acc_list, te_acc_list], ['validate', 'test'],stroe_path=os.path.join(figure_path,'accuracy.png'))