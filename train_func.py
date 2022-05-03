#train
# import args
import torch
import torch.nn
import torch.optim
from torch.autograd import Variable
from model import HCNN
# from pretrained import pre_alexnet_ML,PredTrianedVGG16
def init_model(model_tag=1):

    if model_tag == 1:
        model = PredTrianedVGG16()
        # model = PredTrianedVGG16().cuda()
    else:
        model = pre_alexnet_ML()
    if torch.cuda.is_available():
        print('using cuda')
        model = model.cuda()
        model.criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        model.criterion = torch.nn.CrossEntropyLoss()
    model.train()
    model.optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate)
    model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=args.lr_update_step, gamma=args.lr_decay)
    return model
def train_model(train_data, epoch, model=None, model_tag=1):
    # 单纯的一次训练过程，所有其他事情交由控制器来做，只返回模型和本次的loss。
    if epoch == 0:
        model = init_model(model_tag)
    else:
        model = model
        model.train()
    cell_num = 0
    correct =0
    mpt1 = []
    for cell, cl in train_data:
        if torch.cuda.is_available():
            cell = cell.cuda()
            cl = cl.cuda()
        else:
            cell = Variable(cell)
            cl = Variable(cl).long()
        out = model.forward(cell)
        _, predicted = torch.max(out.data, 1)
        cell_num += cl.size(0)
        correct += (predicted == cl.data).sum()
       # ou = model.forward(cell)
        loss = model.criterion(out, cl)

        print_loss = loss.data.item()
        print_loss = float(print_loss)
        # print(print_loss)
        mpt1.append(print_loss)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    loss = sum(mpt1)
    tr_acc = round(correct.item()/cell_num,2)
    print('train_acc:',tr_acc)
    # print(loss)
    return model, loss,tr_acc
def validate(model,val_loader):
    model.eval()
    test_total = 0
    correct = 0
    eval_number = 0
    for cell, cl in val_loader:
        if torch.cuda.is_available():
            cell = cell.cuda()
            cl = cl.cuda()
        else:
            # SB things change to cuda to,orrow
            cell = Variable(cell)
            cl = Variable(cl).long()

        outputs = model.forward(cell)
        _, predicted = torch.max(outputs.data, 1)
        # lo#ss = cirterion(outputs, labels)
        # test_loss += loss.item()
        test_total += cl.size(0)
        correct += (predicted == cl.data).sum()
        eval_number += correct.item()
    k = correct.item()
    print("correct_item",correct.item())
    # print("eval_number",eval_number)
    accuracy = k / test_total
    return accuracy
