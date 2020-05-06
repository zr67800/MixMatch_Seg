##update log Apr 03:
# use separate net for pseudo label prediction. potentially support ema
# pseudo label guessing done in .train() mode

import os
import sys
import time
import random
import numpy as np
import h5py
import itertools

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from networks.vnet import VNet
from utils import ramps, losses
from dataloaders.la_heart import RandomCrop, CenterCrop, RandomRotFlip
from debugging import save_as_image


GPU = True
TIMETEST = False

DEBUG = False

LABELLED_INDEX = list(range(16))
UNLABELLED_INDEX = list(range(16,80))

if DEBUG:
    c = itertools.count()


def augmentation(volume, aug_factor):
    #volume is numpy array of shape (C, D, H, W)
    return volume + aug_factor * np.clip(np.random.randn(*volume.shape) * 0.1, -0.2, 0.2).astype(np.float32)


def mix_match(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor): 
    # X is labeled data of size BATCH_SIZE, and U is unlabeled data
    # X is list of tuples (data, label), and U is list of data
    # where data and label are of shape (C, D, H, W), numpy array. C of data is 1 and C of label is 2 (one hot)


    b = len(X)

    #step 1: Augmentation
    X_cap = [(augmentation(x[0], aug_factor),x[1]) for x in X ] #shape unchanged
    #U_cap = [[augmentation(u, aug_factor) for i in range(K)] for u in U] #U_cap is a list (length b) of list (length K)
    U = torch.from_numpy(np.array(U)) #[b, 1, D, H, W]
    if GPU:
        U = U.cuda()
    U_cap = U.repeat(K,1,1,1,1) #[K*b, 1, D, H, W]
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2) #augmented.
    
    #step 2: label guessing
    with torch.no_grad():
        Y_u = eval_net(U_cap)
        Y_u = F.softmax(Y_u, dim=1)

    guessed = torch.zeros(U.size()).repeat(1,2,1,1,1) #empty label [b, 2, D, H, W]
    if GPU:
        guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i*b:(i+1)*b]
    guessed /= K 

    #sharpening
    guessed = guessed**(1/T)
    guessed = guessed/guessed.sum(dim=1, keepdim=True)
    guessed = guessed.repeat(K,1,1,1,1)
    guessed = guessed.detach().cpu().numpy() #shape [b,2,D,H,W]

    U_cap = U_cap.detach().cpu().numpy()
    
    U_cap = list(zip(U_cap, guessed))

    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    #step 3: MixUp
    #original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]
    
    W = X_cap+U_cap #length = b+b*k
    random.shuffle(W)

    if x_mixup_mode == 'w':
        X_prime = [mix_up(X_cap[i], W[i], alpha) for i in range(b)]
    elif x_mixup_mode == 'x':
        idxs = np.random.permutation(range(b))
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(b*K))[:b]
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(b)]
    elif x_mixup_mode == '_':
        X_prime =  X_cap
    else:
        raise ValueError('wrong mixup_mode')
    
    if u_mixup_mode == 'w':
        U_prime = [mix_up(U_cap[i], W[b+i], alpha) for i in range(b*K)]
    elif u_mixup_mode == 'x':
        idxs = np.random.permutation(range(b*K))%b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(b*K)]
    elif u_mixup_mode == 'u':
        idxs = np.random.permutation(range(b*K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(b*K)]
    elif u_mixup_mode == '_':
        U_prime =  U_cap
    else:
        raise ValueError('wrong mixup_mode')
            
    #if DEBUG:
        #save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
        #save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")

    return X_prime, U_prime
def mix_up(s1, s2, alpha):
    # s1, s2 are tuples(data, label)
    l = np.random.beta(alpha, alpha)
    l = max(l, 1-l)

    x1,p1 = s1
    x2,p2 = s2

    x = l*x1 + (1-l)*x2
    p = l*p1 + (1-l)*p2

    return (x,p)

def data_loader(train_data_path = '../data/2018LA_Seg_Training Set', split = 'train'):
    ''' return list of dicts('image', 'label') of data. image and label are numpy arrays ( d1, d2, d3)'''
    if split not in {'train','test'}:
        raise ValueError("split must be 'train' or 'test'")
    with open(f"{train_data_path}/../{split}.list", 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n','') for item in image_list]
    data = []
    for name in image_list:
        h5f = h5py.File(f"{train_data_path}/{name}/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        sample = {'image': image, 'label': label}
        data.append(sample)
    return data


## Batch loader
def batch_loader(iterable, batch_size):
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []
    if len(b) > 0:
        yield b

def shape_transform(sample):
    '''
    take in sample{'image':ndarray(d1,d2,d3), 'label':ndarray(d1,d2,d3)}
    return sample{'image':ndarray(1,d1,d2,d3), 'label':ndarray(2,d1,d2,d3), one hot label}
    '''
    image = sample['image'][np.newaxis, :,:,:].astype(np.float32)
    label = sample['label'][np.newaxis, :,:,:].astype(np.float32)
    one_hot = np.concatenate((1-label,label), axis = 0)
    sample = {'image': image, 'label': one_hot}
    return sample



def soft_cross_entropy(predicted, target):
    #print(predicted.type(), target.type())
    return -(target * torch.log(predicted)).sum(dim=1).mean()


global_counter = itertools.count()
def copy_params(net, eval_net, decay):
    i = next(global_counter)
    decay = min(1 - 1 / (i+1), decay)
    for eval_param, param in zip(eval_net.parameters(), net.parameters()):
        eval_param.data.mul_(decay).add_((1-decay)*param.data)

def train_epoch(net, eval_net, labelled_data, unlabelled_data, batch_size, supervised_only, optimizer, x_criterion, u_criterion,
                K, T, alpha, mixup_mode, Lambda, aug_factor,decay):
    if DEBUG:
        epoch = next(c)
    net.train()
    epoch_loss = 0.0
    for i, (l_batch, u_batch) in enumerate(zip(batch_loader(labelled_data, batch_size),batch_loader(unlabelled_data, batch_size))):
        #l_batch, u_batch are lists (len = batch_size) of dicts 
        l_image = [sample['image'] for sample in l_batch]
        l_label = [sample['label'] for sample in l_batch]
        u_image = [sample['image'] for sample in u_batch]
        
        X = list(zip(l_image,l_label))   #list of (image, onehot_label) of length = batch_size
        U = u_image                      #list of image of length = batch_size

        if not supervised_only:
            copy_params(net, eval_net, decay)

            X_prime, U_prime = mix_match(X, U, eval_net = eval_net, K = K, T = T, alpha = alpha, mixup_mode = mixup_mode, aug_factor = aug_factor)
            
            net.train()

            X_data = torch.from_numpy(np.array([x[0] for x in X_prime]))
            X_label = torch.from_numpy(np.array([x[1] for x in X_prime]))
            U_data = torch.from_numpy(np.array([x[0] for x in U_prime]))
            U_label = torch.from_numpy(np.array([x[1] for x in U_prime]))

            if DEBUG:
                #save_as_image(X_data.numpy(), f"../debug_output/x_data")
                save_as_image(U_data.numpy(),f"../debug_output/u_data_{epoch}" )
                #save_as_image(X_label[:, [1], :, :, :].numpy(), f"../debug_output/x_label")
                save_as_image(U_label[:, [1], :, :, :].numpy(),f"../debug_output/u_label_{epoch}" )


            if GPU:
                X_data = X_data.cuda()
                X_label = X_label.cuda()
                U_data = U_data.cuda()
                U_label = U_label.cuda()

            
            X = torch.cat((X_data, U_data), 0)
            Y = net(X)
            Y_x = Y[:len(X_data)]
            Y_u = Y[len(X_data):]

            Y_x_softmax = F.softmax(Y_x, dim=1)
            Y_u_softmax = F.softmax(Y_u, dim=1)

            if DEBUG:
                #save_as_image(Y_x_softmax[:, [1], :, :, :].detach().cpu().numpy(), "../debug_output/x_pred")
                save_as_image(Y_u_softmax[:, [1], :, :, :].detach().cpu().numpy(),f"../debug_output/u_pred_{epoch}" )

            loss_x_seg = x_criterion(Y_x_softmax, X_label)
            loss_x_dice = losses.dice_loss(Y_x_softmax[:, 1, :, :, :], X_label[:, 1, :, :, :])
            loss_x = 0.5*(loss_x_seg+loss_x_dice)

            loss_u = u_criterion(Y_u_softmax, U_label)

            loss = loss_x + Lambda * loss_u

            if DEBUG:
                print(loss_x.item(), loss_u.item(), loss.item())

        else:
            #supervised_only
            X_data = torch.from_numpy(np.array(l_image))
            X_label = torch.from_numpy(np.array(l_label))
            if DEBUG:
                save_as_image(X_data.numpy(), "../debug_output/s_data")
                save_as_image(X_label[:, [1], :, :, :].numpy(), "../debug_output/s_label")
            if GPU:
                X_data = X_data.cuda()
                X_label = X_label.cuda()
            
            Y_x = net(X_data)
            Y_x_softmax = F.softmax(Y_x, dim=1)
            
            if DEBUG:
                save_as_image(Y_x_softmax[:, [1], :, :, :].detach().cpu().numpy(), "../debug_output/s_pred")

            loss_x_seg = x_criterion(Y_x_softmax, X_label)
            loss_x_dice = losses.dice_loss(Y_x_softmax[:, 1, :, :, :], X_label[:, 1, :, :, :])
            loss = 0.5*(loss_x_seg+loss_x_dice)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        if DEBUG:
            break
    return epoch_loss/(i+1)

def validation(net, testing_data, x_criterion):
    net.eval()
    val_dice_loss = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for i, data in enumerate(batch_loader(testing_data, 4)):
            image = [sample['image'] for sample in data]
            label = [sample['label'] for sample in data]
            
            image = torch.from_numpy(np.array(image))
            label = torch.from_numpy(np.array(label))

            if GPU:
                image = image.cuda()
                label = label.cuda()

            Y = net(image)
            Y_softmax = F.softmax(Y, dim=1)
        
            val_dice_loss += losses.dice_loss(Y_softmax[:, 1, :, :, :], label[:, 1, :, :, :]).item()
            
            predictions = Y_softmax.argmax(dim=1, keepdim=True).view_as(label[:, 1, :, :, :])
            accuracy += predictions.eq(label[:, 1, :, :, :].long()).sum().item()/label.sum()
            #print (label.shape, label.sum())

      
    val_dice_loss /= (i+1)
    accuracy /= (i+1)
    return val_dice_loss, accuracy

def linear_ramp(maximum_lambda):
    '''
    ramp up from step 100 to step 200
    '''
    def func(e):
        if e <100:
            return 0
        else:
            return min(maximum_lambda,(e-100)*0.01*maximum_lambda)
    
    return func

def slow_linear_ramp(maximum_lambda):
    '''
    ramp up from step 100 to step 600
    '''
    def func(e):
        if e <100:
            return 0
        else:
            return min(maximum_lambda,(e-100)*0.002*maximum_lambda)
    
    return func


def experiment(exp_identifier, max_epoch, training_data, testing_data, batch_size = 2, supervised_only = False, 
                K = 2, T = 0.5, alpha = 1, mixup_mode = 'all', Lambda = 1, Lambda_ramp = None, base_lr = 0.01, change_lr = None, aug_factor = 1, from_saved = None, always_do_validation = True, decay = 0):
    '''
    max_epoch: epochs to run. Going through labeled data once is one epoch.
    batch_size: batch size of labeled data. Unlabeled data is of the same size.
    training_data: data for train_epoch, list of dicts of numpy array.
    training_data: data for validation, list of dicts of numpy array.
    supervised_only: if True, only do supervised training on LABELLED_INDEX; otherwise, use both LABELLED_INDEX and UNLABELLED_INDEX
    
    Hyperparameters
    ---------------
    K: repeats of each unlabelled data
    T: temperature of sharpening
    alpha: mixup hyperparameter of beta distribution
    mixup_mode: how mixup is performed --
        '__': no mix up
        'ww': x and u both mixed up with w(x+u)
        'xx': both with x
        'xu': x with x, u with u
        'uu': both with u
        ... _ means no, x means with x, u means with u, w means with w(x+u)
    Lambda: loss = loss_x + Lambda * loss_u, relative weight for unsupervised loss
    base_lr: initial learning rate

    Lambda_ramp: callable or None. Lambda is ignored if this is not None. In this case,  Lambda = Lambda_ramp(epoch).
    change_lr: dict, {epoch: change_multiplier}


    '''
    print (f"Experiment {exp_identifier}: max_epoch = {max_epoch}, batch_size = {batch_size}, supervised_only = {supervised_only},"
           f"K = {K}, T = {T}, alpha = {alpha}, mixup_mode = {mixup_mode}, Lambda = {Lambda}, Lambda_ramp = {Lambda_ramp}, base_lr = {base_lr}, aug_factor = {aug_factor}.")

    net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    eval_net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)

    if from_saved is not None:
        net.load_state_dict(torch.load(from_saved))

    if GPU:
        net = net.cuda()
        eval_net.cuda()

    ## eval_net is not updating
    for param in eval_net.parameters():
                param.detach_()

    net.train()
    eval_net.train()


    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    x_criterion = soft_cross_entropy #supervised loss is 0.5*(x_criterion + dice_loss)
    u_criterion = nn.MSELoss() #unsupervised loss

    training_losses = []
    testing_losses = []
    testing_accuracy = [] #dice accuracy

    patch_size = (112, 112, 80)

    testing_data = [shape_transform(CenterCrop(patch_size)(sample)) for sample in testing_data]
    t0 = time.time()

    lr = base_lr

    for epoch in range(max_epoch):
        labelled_index = np.random.permutation(LABELLED_INDEX)
        unlabelled_index = np.random.permutation(UNLABELLED_INDEX)[:len(labelled_index)]
        labelled_data = [training_data[i] for i in labelled_index]
        unlabelled_data = [training_data[i] for i in unlabelled_index] #size = 16


        ##data transformation: rotation, flip, random_crop
        labelled_data = [shape_transform(RandomRotFlip()(RandomCrop(patch_size)(sample))) for sample in labelled_data]
        unlabelled_data = [shape_transform(RandomRotFlip()(RandomCrop(patch_size)(sample))) for sample in unlabelled_data]

        if Lambda_ramp is not None:
            Lambda = Lambda_ramp(epoch)
            print(f"Lambda ramp: Lambda = {Lambda}")

        if change_lr is not None:
            if epoch in change_lr:
                lr_ = lr * change_lr[epoch]
                print (f"Learning rate decay at epoch {epoch}, from {lr} to {lr_}")
                lr = lr_
                #change learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_ 

        

        training_loss = train_epoch(net = net,eval_net = eval_net, labelled_data = labelled_data, unlabelled_data = unlabelled_data,
                                    batch_size = batch_size, supervised_only = supervised_only, 
                                    optimizer = optimizer, x_criterion = x_criterion, u_criterion = u_criterion,
                                    K = K, T = T, alpha = alpha, mixup_mode = mixup_mode, Lambda = Lambda, aug_factor = aug_factor, decay = decay)

        training_losses.append(training_loss)

        if always_do_validation or epoch%50 == 0:
            testing_dice_loss, accuracy = validation(net = net, testing_data = testing_data, x_criterion = x_criterion)

        testing_losses.append(testing_dice_loss)
        testing_accuracy.append(accuracy)
        print (f"Epoch {epoch+1}/{max_epoch}, time used: {time.time()-t0:.2f},  training loss: {training_loss:.6f}, testing dice_loss: {testing_dice_loss:.6f}, testing accuracy: {100.0*accuracy:.2f}% ")

    save_path = f"../saved/{exp_identifier}.pth"
    torch.save(net.state_dict(), save_path)
    print(f"Experiment {exp_identifier} finished. Model saved as {save_path}")
    return training_losses, testing_losses, testing_accuracy

def save_logs_to_csv(logs, path):
    np.savetxt(path, logs, delimiter = ',', fmt = '%.6f')
    print(f"Logs saved as {path}")

def main():
    training_data = data_loader(split = 'train')
    testing_data = data_loader(split = 'test')
    print(f"Data loaded. Training: {len(training_data)}, testing: {len(testing_data)}. ")

    standard_lr_schedular = {313:0.1, 625:0.1} ## referenced to UAMT, per 2500 iters lr*=0.1

    experiments_args = [
 
        # #default one repeat
        
        # {'exp_identifier':'N_1_default_ramp_faster3', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        

        # ####==== Apr03: Stage P: Ablation study ====####

        # ##== K ==##
        # {'exp_identifier':'P_K1_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 1, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_K3_1', 'max_epoch':750, 'batch_size': 1, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 3, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_K4_1', 'max_epoch':750, 'batch_size': 1, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 4, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        # ##== T ==##
        # {'exp_identifier':'P_T1_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 1, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_T0_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 1e-5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_T0.2_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.2, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_T0.75_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.75, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        # ##== EMA ==##
        # {'exp_identifier':'P_EMA_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular, 'decay':0.99},
        
        # ##== mixup mode ==##
        # {'exp_identifier':'P_mixup_00_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': '__', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_mixup_w0_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'w_', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_mixup_0w_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': '_w', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_mixup_xx_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'xx', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_mixup_uu_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'uu', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_mixup_xu_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'xu', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        
        # ##== consistency weight ==##
        # {'exp_identifier':'N_l1_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'N_l10_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(10), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        # ## repeat 3 ##
        # ##== K ==##
        # {'exp_identifier':'P_K1_3', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 1, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_K3_3', 'max_epoch':750, 'batch_size': 1, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 3, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_K4_3', 'max_epoch':750, 'batch_size': 1, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 4, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},




        ####===== Apr 21 ====####

        # T0.01 
        # {'exp_identifier':'P_T0.01_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #               'K': 2, 'T': 0.01, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_T0.01_2', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #               'K': 2, 'T': 0.01, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'P_T0.01_3', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #               'K': 2, 'T': 0.01, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(0.1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        # # N l1/l10
        # {'exp_identifier':'N_l1_3', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(1), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        # {'exp_identifier':'N_l10_3', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
        #              'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(10), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        # N l100, slow ramp
        {'exp_identifier':'N_l100S_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
                     'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': slow_linear_ramp(100), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        {'exp_identifier':'N_l100S_2', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
                     'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': slow_linear_ramp(100), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        {'exp_identifier':'N_l100S_3', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
                     'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': slow_linear_ramp(100), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        # N l100
        {'exp_identifier':'N_l100_1', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
                     'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(100), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        {'exp_identifier':'N_l100_2', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
                     'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(100), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},
        {'exp_identifier':'N_l100_3', 'max_epoch':750, 'batch_size': 2, 'training_data': training_data, 'testing_data': testing_data, 'supervised_only': False, 
                     'K': 2, 'T': 0.5, 'alpha': 0.75, 'mixup_mode': 'ww', 'Lambda_ramp': linear_ramp(100), 'base_lr': 0.01, 'change_lr': standard_lr_schedular},

        
        
    ]


    for kwargs in experiments_args:
        try:

            save_logs_to_csv(
                experiment(**kwargs),
                f"../saved/{kwargs['exp_identifier']}.csv")
        except Exception as e:
            print("An error happened", e)
    

##NOTE TODO 
##



def debugger():
        patch_size = (112, 112, 80)
        training_data = data_loader(split = 'train')
        testing_data = data_loader(split = 'test')

        
        x_criterion = soft_cross_entropy #supervised loss is 0.5*(x_criterion + dice_loss)
        u_criterion = nn.MSELoss() #unsupervised loss

        labelled_index = np.random.permutation(LABELLED_INDEX)
        unlabelled_index = np.random.permutation(UNLABELLED_INDEX)[:len(labelled_index)]
        labelled_data = [training_data[i] for i in labelled_index]
        unlabelled_data = [training_data[i] for i in unlabelled_index] #size = 16


        ##data transformation: rotation, flip, random_crop
        labelled_data = [shape_transform(RandomRotFlip()(RandomCrop(patch_size)(sample))) for sample in labelled_data]
        unlabelled_data = [shape_transform(RandomRotFlip()(RandomCrop(patch_size)(sample))) for sample in unlabelled_data]


        net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
        

        
        model_path = "../saved/0_supervised.pth"
        net.load_state_dict(torch.load(model_path))


        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        training_loss = train_epoch(net = net, labelled_data = labelled_data, unlabelled_data = unlabelled_data,
                                    batch_size = 2, supervised_only = True, 
                                    optimizer = optimizer, x_criterion = x_criterion, u_criterion = u_criterion,
                                    K = 1, T = 1, alpha = 1, mixup_mode = "__", Lambda = 0, aug_factor = 0)



        net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
        model_path = "../saved/8_expected_supervised.pth"
        net.load_state_dict(torch.load(model_path))
        
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        training_loss = train_epoch(net = net, labelled_data = labelled_data, unlabelled_data = unlabelled_data,
                                    batch_size = 2, supervised_only = False, 
                                    optimizer = optimizer, x_criterion = x_criterion, u_criterion = u_criterion,
                                    K = 1, T = 1, alpha = 1, mixup_mode = "__", Lambda = 0, aug_factor = 0)

    

if __name__ == "__main__":



    main()

    #if DEBUG:
    #    debugger()
        