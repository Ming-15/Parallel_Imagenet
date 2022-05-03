import os
import pickle
import torch
def save_pkl(path,save_object):
    # every time init append the obejct to the list.
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            save_objects_list = [save_object]
            pickle.dump(save_objects_list, f)
    else:
        with open(path, 'rb') as f:
            save_objects_list = pickle.load(f)
            save_objects_list.append(save_object)
            with open(path, 'wb') as f:
                pickle.dump(save_objects_list, f)

def visual_progross(it):
    lft = '-' * it
    lrt = ' ' * (100 - it)
    lt = lft + lrt
    print('epoch-execution' + str(it).zfill(2) + '%:' + lt)

def easy_validate(model,val_data):
    with torch.no_grad():
        model.eval()  # prep model for evaluation
        test_total = 0
        correct = 0
        for cell, cl in val_data:
            cell = cell.cuda()
            cl = cl.cuda()
            outputs = model.forward(cell)
            _, predicted = torch.max(outputs.data, 1)
            test_total += cl.size(0)
            correct += (predicted == cl.data).sum()
        k = correct.item()
        accuracy = k / test_total
        print(accuracy, 'validation_accuracy')
    return accuracy

def validate(model,val_data,criterion,valid_losses):
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
        print(accuracy, 'validation_accuracy')
    return accuracy