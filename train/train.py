import argparse
import functools
import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from GhoDenNet import *
from torch.optim.lr_scheduler import StepLR
from reader import CustomDataset
from utility import add_arguments, print_arguments
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    128                  )
add_arg('num_workers',      int,    1                    )
add_arg('num_epoch',        int,    50                   )
add_arg('num_classes',      int,    10                   )
add_arg('learning_rate',    float,  1e-4                 )
add_arg('input_shape',      str,    '(None, 1, 128, 192)')
add_arg('train_list_path',  str,    'data/esc10-train.npy')
add_arg('test_list_path',   str,    'data/esc10-test.npy')
add_arg('save_model',       str,    'models/')
args = parser.parse_args()


def test(model, test_loader, device):
    model.eval()
    accuracies1 = []
    accuracies2 = []

    for batch_id, (spec_mag, label, labelbin) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.to(device).long()
        labelbin = labelbin.to(device).long()
        output1,output2 = model(spec_mag)
        output1 = output1.data.cpu().numpy()
        output1 = np.argmax(output1, axis=1)
        label = label.data.cpu().numpy()
        labelbin = labelbin.data.cpu().numpy()
        acc = np.mean((output1 == labelbin).astype(int))
        accuracies1.append(acc.item())
        output2 = output2.data.cpu().numpy()
        output2 = np.argmax(output2, axis=1)
        acc = np.mean((output2 == label).astype(int))
        accuracies2.append(acc.item())

    model.train()
    acc1 = float(sum(accuracies1) / len(accuracies1))
    acc2 = float(sum(accuracies2) / len(accuracies2))
    return acc1,acc2


def train(args):
    input_shape = eval(args.input_shape)
    train_dataset = CustomDataset(args.train_list_path, model='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_dataset = CustomDataset(args.test_list_path, model='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GhoDenNet()
    model.to(device)  


    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=5e-4)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)  # args.learning_rate


    loss = torch.nn.CrossEntropyLoss()


    for epoch in range(args.num_epoch):
        loss_sum1 = []
        loss_sum2 = []
        accuracies_1 = []
        accuracies_2 = []
        for batch_id, (spec_mag, label, labelbin) in enumerate(train_loader):
            spec_mag = spec_mag.to(device)
            label = label.to(device).long()
            labelbin = labelbin.to(device).long()
            output_1,output_2 = model(spec_mag)

            los1 = loss(output_1, labelbin)
            los2 = loss(output_2, label)
            los = 0.1*los1+los2
            los.backward() 
            optimizer.step() 
 
            output_1 = torch.nn.functional.softmax(output_1)
            output_1 = output_1.data.cpu().numpy()
            output_1 = np.argmax(output_1, axis=1)
            labelbin = labelbin.data.cpu().numpy()
            acc = np.mean((output_1 == labelbin).astype(int))
            accuracies_1.append(acc)

            output_2 = torch.nn.functional.softmax(output_2)
            output_2 = output_2.data.cpu().numpy()
            output_2 = np.argmax(output_2, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output_2 == label).astype(int))
            accuracies_2.append(acc)


            loss_sum1.append(los1)
            loss_sum2.append(los2)


            if batch_id == 3: 
                train_losses1 = sum(loss_sum1) / len(loss_sum1)
                train_losses2 = sum(loss_sum2) / len(loss_sum2)
                train_acc1 = sum(accuracies_1) / len(accuracies_1)
                train_acc2 = sum(accuracies_2) / len(accuracies_2)
                print('[%s] Train epoch %d, batch: %d/%d, loss1: %f,loss2: %f, accuracy1: %f,accuracy2: %f' % (
                    datetime.now(), epoch, batch_id, len(train_loader), train_losses1,train_losses2,
                    train_acc1,train_acc2))

        scheduler.step() 

        if epoch > 8:
            accs1,accs2 = test(model, test_loader, device)

            print('[%s] Test %d, accuracy2: %f, accuracy2: %f' % (datetime.now(), epoch, accs1, accs2))

        # model_path = os.path.join(args.save_model, 'ghostnet.pth')
        # if not os.path.exists(os.path.dirname(model_path)):
        #     os.makedirs(os.path.dirname(model_path))
        # torch.jit.save(torch.jit.script(model), model_path)


if __name__ == '__main__':
    print_arguments(args)
    train(args)


