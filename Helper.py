import os, math, time, copy, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from collections import defaultdict
from abc import ABC, abstractmethod

import tqdm
import torch
import torchvision.transforms as transforms

from torch.autograd import grad
from torchvision import transforms
from torchvision.datasets import CocoDetection, CocoCaptions
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from Loader import load_ihdp, log, load_mimic
from Models import PModel, VModel
from Models import FTransformer
# from MimicModel import FTransformer
from Utils import validation_split
from Optimization import learn_rate_adaptation

rng = np.random.default_rng(0)

def get_transform_cub(size):
    scale = 256.0/size
    target_resolution = (size, size)
    center_transforms= [
        transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
        transforms.CenterCrop(target_resolution)
    ]
    tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    augmentation_list = center_transforms + tensor_transforms
    transform = transforms.Compose(augmentation_list)
    return transform

# class MSCoco(CocoDetection):
#     def __init__(self, root, annFile, transforms):
#         super(CocoDetection, self).__init__()

class Trainer(ABC):
    @abstractmethod
    def __init__(self, model, optimizer, scheduler, D_train, D_test, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.D_train = D_train
        self.D_test = D_test
        self.criterion = loss_fn
        self.trainloss = defaultdict(list)
        self.valloss = defaultdict(list)
    
    @abstractmethod
    def train(self, root: str, n_epoches: int, device: str):
        pass

class FTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, D_train, D_test, loss_fn, n_exp=10, outname='trainmodel_ft'):
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, D_train=D_train, D_test=D_test, loss_fn=loss_fn)
        self.n_exp = n_exp
        self.anomaly_list = [8, 9, 12, 13, 23, 25, 27, 28, 33, 52, 67, 80, 83, 84, 85, 92, 97]
        self.init_lr = optimizer.param_groups[0]['lr']
        self.outname = outname

    def train(self, root='C:\\Workspace\\', n_epoches=10, device='cpu'):
        self.model.to(device)
        d_model = self.model.d_model
        self.model.class_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_mse = 10000
        best_cate = 10000

        global_epoch = 0
        current_epoch = 0
        global_step = 0
        total_train_mse = []
        total_val_mse = []
        total_train_ate = []
        total_val_ate = []
        total_train_cate = []
        total_val_cate = []

        for j in range(self.n_exp):
            print("exp: " + str(j) + '===================================================')
            D_exp = self.D_train[j]
            n = D_exp['x'].shape[0]
            pt = np.mean(D_exp['t'])
            print(f'propensity: {(pt*100):>0.1f}%')
            batch_size = 32

            exp_train_mse = []
            exp_val_mse = []
            exp_train_ate = []
            exp_val_ate = []
            exp_train_cate = []
            exp_val_cate = []

            for epoch in range(n_epoches):
                print(("Epoch: %f/%f" % (epoch, n_epoches - 1)))
                print(("----------"))
                current_epoch+=1
                
                I_train, I_valid = validation_split(n, 0.2)
                ''' Train/validation split '''
                n_train = len(I_train)
                n_val = len(I_valid)
                I = list(range(0, n_train))
                
                def softclip(tensor, max):
                    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                    result_tensor = tensor - torch.nn.softplus(tensor - max)
                    return result_tensor

                running_loss = 0.0
                running_val_loss = 0.0
                running_atebias = 0.0
                running_val_atebias = 0.0
                running_catebias = 0.0
                running_val_catebias = 0.0

                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()
                        for i_batch in range(n_train // batch_size):
                            if i_batch < (n_train // batch_size - 1):
                                I_b = I[i_batch * batch_size:(i_batch+1) * batch_size]
                            else:
                                I_b = I[i_batch * batch_size:]
                            x_batch = D_exp['x'][I_train,:][I_b,:]
                            x_batch_ = RandShuffle(x_batch)
                            x_batch = torch.tensor(x_batch_).to(torch.float32).to(device)

                            t_batch = D_exp['t'][I_train,:][I_b]
                            y_batch = D_exp['yf'][I_train,:][I_b]
                            ycf_batch = D_exp['ycf'][I_train,:][I_b]
                            mu0_batch = D_exp['mu0'][I_train,:][I_b]
                            mu1_batch = D_exp['mu1'][I_train,:][I_b]

                            x_batch = torch.tensor(x_batch).to(torch.float32).to(device)
                            t_batch = torch.tensor(t_batch).to(device)
                            y_batch = torch.tensor(y_batch).to(device)
                            ycf_batch = torch.tensor(ycf_batch).to(device)
                            mu0_batch = torch.tensor(mu0_batch).to(device)
                            mu1_batch = torch.tensor(mu1_batch).to(device)

                            with torch.set_grad_enabled(True):
                                self.optimizer.zero_grad()
                                yf, ycf = self.model(x_batch, t_batch)
                                loss_value = torch.nn.functional.mse_loss(yf, y_batch)
                                loss_value.backward()
                                # print(loss_value.numpy())
                                running_loss += loss_value * t_batch.size(0)
                                self.trainloss['mse'].append(loss_value.item())

                                self.optimizer.step()
                                self.scheduler.step(global_step)
                                global_step += 1

                                eff_pred = yf - ycf
                                eff_pred = torch.where(t_batch>0, eff_pred, -eff_pred)
                                ate_pred = torch.mean(eff_pred)

                                eff_batch = y_batch - ycf_batch
                                eff_batch = torch.where(t_batch>0, eff_batch, -eff_batch)
                                ate_batch = torch.mean(eff_batch)

                                ate_bias = ate_pred - ate_batch

                                cate_batch = mu0_batch - mu1_batch
                                cate_bias = eff_pred - cate_batch
                                cate_bias = torch.sqrt(torch.mean(torch.square(cate_bias)))

                                self.trainloss['ate'].append(ate_bias.item())
                                running_atebias += ate_bias * t_batch.size(0)
                                self.trainloss['cate'].append(cate_bias.item())
                                running_catebias += cate_bias**2 * t_batch.size(0)
                    else:
                        self.model.eval()
                        I = list(range(0, n_val))
                        for i_batch in range(n_val // batch_size):
                            if i_batch < (n_val // batch_size - 1):
                                I_b = I[i_batch * batch_size:(i_batch+1) * batch_size]
                            else:
                                I_b = I[i_batch * batch_size:]
                            x_batch = D_exp['x'][I_valid,:][I_b,:]
                            t_batch = D_exp['t'][I_valid,:][I_b]
                            y_batch = D_exp['yf'][I_valid,:][I_b]
                            ycf_batch = D_exp['ycf'][I_valid,:][I_b]
                            mu0_batch = D_exp['mu0'][I_valid,:][I_b]
                            mu1_batch = D_exp['mu1'][I_valid,:][I_b]

                            x_batch = torch.tensor(x_batch).to(torch.float32).to(device)
                            t_batch = torch.tensor(t_batch).to(device)
                            y_batch = torch.tensor(y_batch).to(device)
                            ycf_batch = torch.tensor(ycf_batch).to(device)
                            mu0_batch = torch.tensor(mu0_batch).to(device)
                            mu1_batch = torch.tensor(mu1_batch).to(device)

                            with torch.set_grad_enabled(False):
                                yf, ycf = self.model(x_batch, t_batch)
                                loss_value = torch.nn.functional.mse_loss(yf, y_batch)
                                running_val_loss += loss_value * t_batch.size(0)
                                self.valloss['mse'].append(loss_value.item())

                                eff_pred = yf - ycf
                                eff_pred = torch.where(t_batch>0, eff_pred, -eff_pred)
                                ate_pred = torch.mean(eff_pred)

                                eff_batch = y_batch - ycf_batch
                                eff_batch = torch.where(t_batch>0, eff_batch, -eff_batch)
                                ate_batch = torch.mean(eff_batch)

                                ate_bias = ate_pred - ate_batch

                                cate_batch = mu0_batch - mu1_batch
                                cate_bias = eff_pred - cate_batch
                                cate_bias = torch.sqrt(torch.mean(torch.square(cate_bias)))

                                self.valloss['ate'].append(ate_bias.item())
                                running_val_atebias += ate_bias * t_batch.size(0)
                                self.valloss['cate'].append(cate_bias.item())
                                running_val_catebias += cate_bias**2 * t_batch.size(0)

                    if phase == 'train':
                        epoch_loss = running_loss / n_train
                        epoch_ate = running_atebias / n_train
                        epoch_cate = torch.sqrt(running_catebias / n_train)
                        exp_train_mse.append(epoch_loss.item())
                        exp_train_ate.append(epoch_ate.item())
                        exp_train_cate.append(epoch_cate.item())
                        print(f'train loss: {epoch_loss:>8f}, train ate bias: {epoch_ate:>8f}, train cate bias: {epoch_cate:>8f}.')

                    if phase == 'val':
                        epoch_val_loss = running_val_loss / n_val
                        epoch_val_ate = running_val_atebias / n_val
                        epoch_val_cate = torch.sqrt(running_val_catebias / n_val)
                        exp_val_mse.append(epoch_val_loss.item())
                        exp_val_ate.append(epoch_val_ate.item())
                        exp_val_cate.append(epoch_val_cate.item())
                        print(f'val loss: {epoch_val_loss:>8f}, val ate bias: {epoch_val_ate:>8f}, val cate bias: {epoch_val_cate:>8f}')

            total_train_mse.append(exp_train_mse)
            total_val_mse.append(exp_val_mse)
            total_train_ate.append(exp_train_ate)
            total_val_ate.append(exp_val_ate)
            total_train_cate.append(exp_train_cate)
            total_val_cate.append(exp_val_cate)

            mean_exp_val_mse = np.mean(exp_val_mse)
            # if mean_exp_val_mse < best_mse:
            #     best_mse = mean_exp_val_mse
            #     best_model_wts = copy.deepcopy(self.model.state_dict())
            #     global_epoch =  current_epoch
            mean_exp_train_mse = np.mean(exp_train_mse)
            mean_exp_train_ate = np.mean(exp_train_ate)
            mean_exp_train_cate = np.mean(exp_train_cate)
            mean_exp_val_ate = np.mean(exp_val_ate)
            mean_exp_val_cate = np.mean(exp_val_cate)
            print(f'mean exp train loss: {mean_exp_train_mse:>8f}, mean exp train ate bias: {mean_exp_train_ate:>8f}, mean exp train cate bias: {mean_exp_train_cate:>8f}.')
            print(f'mean exp val loss: {mean_exp_val_mse:>8f}, mean exp val ate bias: {mean_exp_val_ate:>8f}, mean exp val cate bias: {mean_exp_val_cate:>8f}')

        self.trainloss['epoch'].extend(np.mean(total_train_mse, axis=0))
        self.trainloss['exp'].extend(np.mean(total_train_mse, axis=1))
        self.trainloss['epoch_ate'].extend(np.mean(total_train_ate, axis=0))
        self.trainloss['exp_ate'].extend(np.mean(total_train_ate, axis=1))
        self.trainloss['epoch_cate'].extend(np.mean(total_train_cate, axis=0))
        self.trainloss['exp_cate'].extend(np.mean(total_train_cate, axis=1))

        self.valloss['epoch'].extend(np.mean(total_val_mse, axis=0))
        self.valloss['exp'].extend(np.mean(total_val_mse, axis=1))
        self.valloss['epoch_ate'].extend(np.mean(total_val_ate, axis=0))
        self.valloss['exp_ate'].extend(np.mean(total_val_ate, axis=1))
        self.valloss['epoch_cate'].extend(np.mean(total_val_cate, axis=0))
        self.valloss['exp_cate'].extend(np.mean(total_val_cate, axis=1))

        time_elapsed = time.time() - since
        print(("Training complete in %sm %ss" % (time_elapsed // 60, time_elapsed % 60)))

        # self.model.load_state_dict(best_model_wts)
        torch.save({
            'epoch': global_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'trainloss': self.trainloss,
            'valloss': self.valloss,
            'best val mse': best_mse,
            }, root+self.outname)
        return self.model
    
class Evaluator:
    def __init__(self, model, dataset, loss_fn, trainloss, valloss, n_exp: int=100):
        self.model = model
        self.D_test = dataset
        self.criterion = loss_fn
        self.dataset = dataset
        self.trainloss = trainloss
        self.valloss = valloss
        self.testloss = defaultdict(list)
        self.n_exp = n_exp

    def evaluate(self, device='cpu'):
        self.model.to(device)
        self.model.eval()
        for j in range(self.n_exp):
            print("exp: " + str(j) + '===================================================')

            D_exp = self.D_test[j]
            n = D_exp['x'].shape[0]
            I = list(range(0, n))
            pt = np.mean(D_exp['t'])
            print(f'propensity: {(pt*100):>0.1f}%')
            batch_size = 32

            def softclip(tensor, max):
                """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                result_tensor = tensor - torch.nn.softplus(tensor - max)
                return result_tensor

            running_loss = 0.0
            running_atebias = 0.0
            running_catebias = 0.0

            for i_batch in range(n // batch_size):
                if i_batch < (n // batch_size - 1):
                    I_b = I[i_batch * batch_size:(i_batch+1) * batch_size]
                else:
                    I_b = I[i_batch * batch_size:]
                x_batch = D_exp['x'][I_b,:]
                t_batch = D_exp['t'][I_b]
                y_batch = D_exp['yf'][I_b]
                ycf_batch = D_exp['ycf'][I_b]
                mu0_batch = D_exp['mu0'][I_b]
                mu1_batch = D_exp['mu1'][I_b]

                x_batch = torch.tensor(x_batch).to(torch.float32).to(device)
                t_batch = torch.tensor(t_batch).to(device)
                y_batch = torch.tensor(y_batch).to(device)
                ycf_batch = torch.tensor(ycf_batch).to(device)
                mu0_batch = torch.tensor(mu0_batch).to(device)
                mu1_batch = torch.tensor(mu1_batch).to(device)

                with torch.set_grad_enabled(False):
                    yf, ycf = self.model(x_batch, t_batch)
                    loss_value = torch.nn.functional.mse_loss(yf, y_batch)
                    running_loss += loss_value * t_batch.size(0)
                    self.testloss['mse'].append(loss_value.item())

                    eff_pred = yf - ycf
                    eff_pred = torch.where(t_batch>0, eff_pred, -eff_pred)
                    ate_pred = torch.mean(eff_pred)

                    eff_batch = y_batch - ycf_batch
                    eff_batch = torch.where(t_batch>0, eff_batch, -eff_batch)
                    ate_batch = torch.mean(eff_batch)

                    ate_bias = ate_pred - ate_batch

                    cate_batch = mu0_batch - mu1_batch
                    cate_bias = eff_pred - cate_batch
                    cate_bias = torch.sqrt(torch.mean(torch.square(cate_bias)))

                    self.testloss['ate'].append(ate_bias.item())
                    running_atebias += ate_bias * t_batch.size(0)
                    self.testloss['cate'].append(cate_bias.item())
                    running_catebias += cate_bias**2 * t_batch.size(0)
            
            epoch_test_loss = running_loss / n
            epoch_test_ate = running_atebias / n
            epoch_test_cate = torch.sqrt(running_catebias / n)
            self.testloss['exp'].append(epoch_test_loss.item())
            self.testloss['exp_ate'].append(epoch_test_ate.item())
            self.testloss['exp_cate'].append(epoch_test_cate.item())
            print(f'loss: {epoch_test_loss:>8f}, ate bias: {epoch_test_ate:>8f}, cate bias: {epoch_test_cate:>8f}')
        print('summary')
        print('test mse: %f, test ate bias: %f, test cate bias: %f' % (np.mean(self.testloss['exp']), np.mean(self.testloss['exp_ate']), np.mean(self.testloss['exp_cate'])))


    def random_sample(self):
        testset = self.dataset
        random_images = []
        m = torch.nn.Softmax(dim=1)
        
        for i in range(0, 15):
            r = random.randint(1, len(testset))
            sample = testset[r]
            ypred = self.model(sample[0].unsqueeze(0))
            prob = m(ypred)
            yi = torch.argmax(prob[0]).item()
            yp = torch.max(prob[0]).item()
            random_images.append((sample[0], f"label:{sample[1]}, pred:{yi}, prob:{yp:.4f}"))
        
        fig, axs = plt.subplots(5, 3, figsize=(28, 28), layout='constrained')
        i = 0
        ims = []
        for ax, image in zip(axs.flat, random_images):
            if ax is None:
                ax = plt.gca()
            im = ax.imshow(np.transpose(image[0], (1, 2, 0)), cmap=plt.cm.gray)
            ims.append(im)
            ax.set_title(image[1])
            i+=1
        plt.show()
    
    def result_vis(self):
        records = [('train mse', self.trainloss['mse']),
                   ('train epoch mse', self.trainloss['exp']),
                   ('train exp mse', self.trainloss['epoch']),
                   ('train ate', self.trainloss['ate']),
                   ('train epoch ate', self.trainloss['exp_ate']),
                   ('train exp ate', self.trainloss['epoch_ate']),
                   ('train cate', self.trainloss['cate']),
                   ('train epoch cate', self.trainloss['exp_cate']),
                   ('train exp cate', self.trainloss['epoch_cate']),
                   ('val mse', self.valloss['mse']),
                   ('val epoch mse', self.valloss['exp']),
                   ('val exp mse', self.valloss['epoch']),
                   ('val ate', self.valloss['ate']),
                   ('val epoch ate', self.valloss['exp_ate']),
                   ('val exp ate', self.valloss['epoch_ate']),
                   ('val cate', self.valloss['cate']),
                   ('val epoch cate', self.valloss['exp_cate']),
                   ('val exp cate', self.valloss['epoch_cate']),
                   ('test mse', self.testloss['mse']),
                   ('test exp mse', self.testloss['exp']),
                   ('test ate', self.testloss['ate']),
                   ('test exp ate', self.testloss['exp_ate']),
                   ('test cate', self.testloss['cate']),
                   ('test exp cate', self.testloss['exp_cate']),
                   ]
        fig, axs = plt.subplots(8, 3, figsize=(32, 32), layout='constrained')
        ii = 0
        clist = [6, 7, 8, 15, 16, 17, 22, 23]
        for ax, rec in zip(axs.flat, records):
            if ax is None:
                ax = plt.gca()
            # im = ax.plot(rec[1], c='b', alpha=0.8)
            xx = 0.5 + np.arange(len(rec[1]))
            # if ii not in clist:
                # im = ax.fill_between(xx, 0, rec[1], alpha=0.5)
            # else:
            im = ax.plot(rec[1], c='b', alpha=0.5)
            ax.set_title(rec[0])
        plt.show()
    
    def result_pre_vis(self):
        records = [('train bce', self.trainloss['bce']),
                   ('train exp bce', self.trainloss['exp']),
                   ('train epoch bce', self.trainloss['epoch']),
                   ('train mlm', self.trainloss['mlm']),
                   ('train exp mlm', self.trainloss['exp mlm']),
                   ('train epoch mlm', self.trainloss['epoch mlm']),
                   ('val bce', self.valloss['bce']),
                   ('val exp bce', self.valloss['exp']),
                   ('val epoch bce', self.valloss['epoch']),
                   ('val mlm', self.valloss['mlm']),
                   ('val exp mlm', self.valloss['exp mlm']),
                   ('val epoch mlm', self.valloss['epoch mlm']),
                   ]
        fig, axs = plt.subplots(4, 3, figsize=(32, 32), layout='constrained')
        ii = 0
        clist = [6, 7, 8, 15, 16, 17, 22, 23]
        for ax, rec in zip(axs.flat, records):
            if ax is None:
                ax = plt.gca()
            # im = ax.plot(rec[1], c='b', alpha=0.8)
            xx = 0.5 + np.arange(len(rec[1]))
            # if ii not in clist:
                # im = ax.fill_between(xx, 0, rec[1], alpha=0.5)
            # else:
            im = ax.plot(rec[1], c='b', alpha=0.5)
            ax.set_title(rec[0])
        plt.show()


class TTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, D_train, D_test, loss_fn, outname='Pmodel', n_exp=100):
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, D_train=D_train, D_test=D_test, loss_fn=loss_fn)
        self.outname = outname
        self.n_exp = n_exp

    def train(self, root='C:\\Workspace\\', n_epoches=10, device='cpu'):
        self.model.to(device)
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        global_epoch = 0
        current_epoch = 0
        total_train_loss = []
        total_val_loss = []
        total_train_acc = []
        total_val_acc = []
        for epoch in range(n_epoches):
            print(("Epoch: %f/%f" % (epoch, n_epoches - 1)))
            print(("----------"))
            current_epoch+=1
            exp_train_loss = []
            exp_val_loss = []
            exp_train_acc = []
            exp_val_acc = []
            for j in range(self.n_exp):
                print("exp: " + str(j) + '===================================================')
                D_exp = self.D_train[j]
                n = D_exp['x'].shape[0]
                I_train, I_valid = validation_split(n, 0.2)
                ''' Train/validation split '''
                n_train = len(I_train)
                n_val = len(I_valid)
                I = list(range(0, n_train))
                pt = np.mean(D_exp['t'])
                print(f'propensity: {(pt*100):>0.1f}%')
                batch_size = 32

                def softclip(tensor, max):
                    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                    result_tensor = tensor - torch.nn.softplus(tensor - max)
                    return result_tensor

                running_loss = 0.0
                running_val_loss = 0.0
                running_correct = 0
                running_val_correct = 0
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()
                        for i_batch in range(n_train // batch_size):
                            if i_batch < (n_train // batch_size - 1):
                                I_b = I[i_batch * batch_size:(i_batch+1) * batch_size]
                            else:
                                I_b = I[i_batch * batch_size:]
                            x_batch = D_exp['x'][I_train,:][I_b,:]
                            t_batch = D_exp['t'][I_train,:][I_b]
                            x_batch = torch.tensor(x_batch).to(torch.float32).to(device)
                            t_batch = torch.tensor(t_batch).squeeze().to(torch.float32).to(device)

                            with torch.set_grad_enabled(True):
                                self.optimizer.zero_grad()
                                logits = self.model(x_batch).squeeze()
                                probs = torch.nn.functional.sigmoid(logits)
                                preds = (torch.rand(t_batch.size(0)) < probs).to(torch.int32).to(device)
                                loss_value = self.criterion(logits, t_batch)
                                loss_value.backward()
                                # print(loss_value.numpy())
                                running_loss += loss_value * t_batch.size(0)
                                batch_corrects = torch.sum(preds == t_batch.data)
                                running_correct += batch_corrects
                                batch_acc = batch_corrects / t_batch.size(0)
                                self.trainloss['bce'].append(loss_value.item())
                                self.trainloss['acc'].append(batch_acc.item())
                                self.optimizer.step()
                    else:
                        self.model.eval()
                        I = list(range(0, n_val))
                        for i_batch in range(n_val // batch_size):
                            if i_batch < (n_val // batch_size - 1):
                                I_b = I[i_batch * batch_size:(i_batch+1) * batch_size]
                            else:
                                I_b = I[i_batch * batch_size:]
                            x_batch = D_exp['x'][I_valid,:][I_b,:]
                            t_batch = D_exp['t'][I_valid,:][I_b]
                            x_batch = torch.tensor(x_batch).to(torch.float32).to(device)
                            t_batch = torch.tensor(t_batch).squeeze().to(torch.float32).to(device)

                            with torch.set_grad_enabled(False):
                                logits = self.model(x_batch).squeeze()
                                probs = torch.nn.functional.sigmoid(logits)
                                preds = (torch.rand(t_batch.size(0)) < probs).to(torch.int32).to(device)
                                loss_value = self.criterion(logits, t_batch)
                                running_val_loss += loss_value * t_batch.size(0)
                                batch_corrects = torch.sum(preds == t_batch.data)
                                running_val_correct += batch_corrects
                                batch_acc = batch_corrects / t_batch.size(0)
                                self.valloss['bce'].append(loss_value.item())
                                self.valloss['acc'].append(batch_acc.item())
                    if phase == 'train':
                        epoch_loss = running_loss / n_train
                        epoch_acc = running_correct / n_train
                        exp_train_loss.append(epoch_loss.item())
                        exp_train_acc.append(epoch_acc.item())
                        print(f'train loss: {epoch_loss:>8f}, train acc: {(100*epoch_acc):>0.1f}%.')

                    if phase == 'val':
                        epoch_val_loss = running_val_loss / n_val
                        epoch_val_acc = running_val_correct / n_val
                        exp_val_loss.append(epoch_val_loss.item())
                        exp_val_acc.append(epoch_val_acc.item())
                        print(f'val loss: {epoch_val_loss:>8f}, val acc: {(100*epoch_val_acc):>0.1f}%.')

            total_train_loss.append(exp_train_loss)
            total_train_acc.append(exp_train_acc)
            total_val_loss.append(exp_val_loss)
            total_val_acc.append(exp_val_acc)

            mean_exp_val_acc = np.mean(exp_val_acc)
            if mean_exp_val_acc > best_acc:
                best_acc = mean_exp_val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                global_epoch =  current_epoch
            mean_exp_train_loss = np.mean(exp_train_loss)
            mean_exp_train_acc = np.mean(exp_train_acc)
            mean_exp_val_loss = np.mean(exp_val_loss)
            print(f'mean exp train loss: {mean_exp_train_loss:>8f}, mean exp train acc: {(100*mean_exp_train_acc):>0.1f}%.')
            print(f'mean exp val loss: {mean_exp_val_loss:>8f}, mean exp val acc: {(100*mean_exp_val_acc):>0.1f}%.')

        self.trainloss['epoch'].extend(np.mean(total_train_loss, axis=1))
        self.trainloss['exp'].extend(np.mean(total_train_loss, axis=0))
        self.trainloss['epoch_acc'].extend(np.mean(total_train_acc, axis=1))
        self.trainloss['exp_acc'].extend(np.mean(total_train_acc, axis=0))

        self.valloss['epoch'].extend(np.mean(total_val_loss, axis=1))
        self.valloss['exp'].extend(np.mean(total_val_loss, axis=0))
        self.valloss['epoch_acc'].extend(np.mean(total_val_acc, axis=1))
        self.valloss['exp_acc'].extend(np.mean(total_val_acc, axis=0))

        time_elapsed = time.time() - since
        print(("Training complete in %sm %ss" % (time_elapsed // 60, time_elapsed % 60)))

        self.model.load_state_dict(best_model_wts)
        torch.save({
            'epoch': global_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'trainloss': self.trainloss,
            'valloss': self.valloss,
            'best val mse': best_acc,
            }, root+self.outname)                    
        return best_model_wts

def RandShuffle(x: torch.Tensor):
    pflag = rng.random()
    if pflag < 0.5:
        x_cont = x[:,:6]
        x_cont_ = rng.permutation(x_cont, axis=1)
        x_bin = x[:,6:]
        x_bin_ = rng.permutation(x_bin, axis=1)
        x = np.concatenate((x_cont_, x_bin_), axis=1)
    return x

class PreTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, D_train, D_test, loss_fn, outname='pretrainFT', pmodel=None, n_exp=100):
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, D_train=D_train, D_test=D_test, loss_fn=loss_fn)
        self.outname = outname
        self.n_exp = n_exp
        self.pmodel = pmodel
        self.init_lr = optimizer.param_groups[0]['lr']

    def train(self, root='C:\\Workspace\\', n_epoches=10, device='cpu'):
        self.model.to(device)
        torch._assert(self.pmodel!=None, f"pmodel is not setted!")
        self.pmodel.to(device)
        self.pmodel.eval()
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 100000

        global_step = 0
        global_epoch = 0
        current_epoch = 0
        total_train_loss = []
        total_train_mlm_loss = []
        total_val_loss = []
        total_val_mlm_loss = []

        for epoch in range(n_epoches):
            print(("Epoch: %f/%f" % (epoch, n_epoches - 1)))
            print(("----------"))
            current_epoch+=1

            exp_train_loss = []
            exp_train_mlm_loss = []
            exp_val_loss = []
            exp_val_mlm_loss = []

            for j in range(self.n_exp):
                print("exp: " + str(j) + '===================================================')
                D_exp = self.D_train[j]
                n = D_exp['x'].shape[0]
            
                pt = np.mean(D_exp['t'])
                print(f'data propensity: {(pt*100):>0.1f}%')
                # pt_m = self.pmodel(torch.tensor(D_exp['x']).to(torch.float32).to(device))
                # pt_m = torch.nn.functional.sigmoid(pt_m)
                # print(f'model propensity: {(torch.mean(pt_m).item()*100):>0.1f}%')
                batch_size = 32
            
                I_train_from, I_valid_from = validation_split(n, 0.2)
                # I_train_to = np.random.permutation(I_train_from) 
                # I_valid_to = np.random.permutation(I_valid_from)
                I_train_to = rng.permutation(I_train_from)
                I_valid_to = rng.permutation(I_valid_from)
                ''' Train/validation split '''
                n_train = len(I_train_from)
                n_val = len(I_valid_from)
                I = list(range(0, n_train))

                def softclip(tensor, max):
                    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                    result_tensor = tensor - torch.nn.softplus(tensor - max)
                    return result_tensor

                running_loss = 0.0
                running_mlm_loss = 0.0
                running_val_loss = 0.0
                running_val_mlm_loss = 0.0
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()
                        for i_batch in range(n_train // batch_size):
                            if i_batch < (n_train // batch_size - 1):
                                I_b = I[i_batch * batch_size:(i_batch+1) * batch_size]
                            else:
                                I_b = I[i_batch * batch_size:]
                            x_batch_from = D_exp['x'][I_train_from,:][I_b,:]
                            x_batch_from = torch.tensor(x_batch_from).to(torch.float32).to(device)
                            
                            pflag = rng.random()
                            if pflag < 0.1:
                                x_batch_to = x_batch_from
                            else:
                                x_batch_to = D_exp['x'][I_train_to,:][I_b,:]
                                x_batch_to = torch.tensor(x_batch_to).to(torch.float32).to(device)

                            ps_from = self.pmodel(x_batch_from.detach())
                            ps_prob_from = torch.nn.functional.sigmoid(ps_from)
                            ps_to = self.pmodel(x_batch_to.detach())
                            ps_prob_to = torch.nn.functional.sigmoid(ps_to)
                            # target = torch.cat([abs(ps_prob_from - ps_prob_to), 1-abs(ps_prob_from - ps_prob_to)],dim=1)
                            target = abs(ps_prob_from - ps_prob_to)

                            x_batch_from = D_exp['x'][I_train_from,:][I_b,:]
                            # x_batch_from = RandShuffle(x_batch_from)
                            x_batch_from = torch.tensor(x_batch_from).to(torch.float32).to(device)
                            
                            x_batch_to = D_exp['x'][I_train_to,:][I_b,:]
                            # x_batch_to = RandShuffle(x_batch_to)
                            x_batch_to = torch.tensor(x_batch_to).to(torch.float32).to(device)

                            with torch.set_grad_enabled(True):
                                self.optimizer.zero_grad()
                                logits, masked_lm_loss = self.model.pretrain(x_batch_from, x_batch_to)
                                loss_value = self.criterion(logits, target.detach())
                                # loss_value_ = torch.mean(loss_value)
                                loss = loss_value + masked_lm_loss
                                # loss = loss_value
                                loss.backward()
                                # print(loss_value.numpy())
                                running_loss += loss_value * x_batch_from.size(0)
                                running_mlm_loss += masked_lm_loss * x_batch_from.size(0)
                                self.trainloss['bce'].append(loss_value.item())
                                self.trainloss['mlm'].append(masked_lm_loss.item())
                                
                                global_step += 1
                                lr_adapted = learn_rate_adaptation(init_lr=self.init_lr, global_step=global_step)
                                if lr_adapted !=0:
                                    for group in self.optimizer.param_groups:
                                        group['lr'] = lr_adapted
                                    self.optimizer.step()
                                else:
                                    self.optimizer.step()
                                    self.scheduler.step(global_step)
                    else:
                        self.model.eval()
                        I = list(range(0, n_val))
                        for i_batch in range(n_val // batch_size):
                            if i_batch < (n_val // batch_size - 1):
                                I_b = I[i_batch * batch_size:(i_batch+1) * batch_size]
                            else:
                                I_b = I[i_batch * batch_size:]
                            x_batch_from = D_exp['x'][I_valid_from,:][I_b,:]
                            x_batch_from = torch.tensor(x_batch_from).to(torch.float32).to(device)
                            x_batch_to = D_exp['x'][I_valid_to,:][I_b,:]
                            x_batch_to = torch.tensor(x_batch_to).to(torch.float32).to(device)
                            ps_from = self.pmodel(x_batch_from.detach())
                            ps_prob_from = torch.nn.functional.sigmoid(ps_from)
                            ps_to = self.pmodel(x_batch_to.detach())
                            ps_prob_to = torch.nn.functional.sigmoid(ps_to)
                            # target = torch.cat([abs(ps_prob_from - ps_prob_to), 1-abs(ps_prob_from - ps_prob_to)],dim=1)
                            target = abs(ps_prob_from - ps_prob_to)

                            with torch.set_grad_enabled(False):
                                logits, masked_lm_loss = self.model.pretrain(x_batch_from, x_batch_to)
                                loss_value = self.criterion(logits, target)
                                running_val_loss += loss_value * x_batch_from.size(0)
                                running_val_mlm_loss += masked_lm_loss * x_batch_from.size(0)
                                # loss_value_ = torch.mean(loss_value)
                                loss = loss_value + masked_lm_loss
                                self.valloss['bce'].append(loss_value.item())
                                self.valloss['mlm'].append(masked_lm_loss.item())

                    if phase == 'train':
                        epoch_loss = running_loss / n_train
                        epoch_mlm_loss = running_mlm_loss / n_train
                        exp_train_loss.append(epoch_loss.item())
                        exp_train_mlm_loss.append(epoch_mlm_loss.item())
                        print(f'train loss: {epoch_loss:>8f}, train mlm loss: {epoch_mlm_loss:>8f}.')

                    if phase == 'val':
                        epoch_val_loss = running_val_loss / n_val
                        epoch_val_mlm_loss = running_val_mlm_loss / n_val 
                        exp_val_loss.append(epoch_val_loss.item())
                        exp_val_mlm_loss.append(epoch_val_mlm_loss.item())
                        print(f'val loss: {epoch_val_loss:>8f}, val mlm loss: {epoch_val_mlm_loss}.')

            total_train_loss.append(exp_train_loss)
            total_val_loss.append(exp_val_loss)
            total_train_mlm_loss.append(exp_train_mlm_loss)
            total_val_mlm_loss.append(exp_val_mlm_loss)

            mean_exp_val_loss = np.mean(exp_val_loss)
            if best_loss > mean_exp_val_loss:
                best_loss = mean_exp_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                global_epoch =  current_epoch
            mean_exp_train_loss = np.mean(exp_train_loss)
            mean_exp_train_mlm_loss = np.mean(exp_train_mlm_loss)
            mean_exp_val_mlm_loss = np.mean(exp_val_mlm_loss)
            print(f'mean exp train loss: {mean_exp_train_loss:>8f}, mean exp train mlm loss: {mean_exp_train_mlm_loss:>8f}.')
            print(f'mean exp val loss: {mean_exp_val_loss:>8f}, mean exp val mlm loss: {mean_exp_val_mlm_loss:>8f}.')

        self.trainloss['epoch'].extend(np.mean(total_train_loss, axis=1))
        self.trainloss['exp'].extend(np.mean(total_train_loss, axis=0))
        self.trainloss['epoch mlm'].extend(np.mean(total_train_mlm_loss, axis=1))
        self.trainloss['exp mlm'].extend(np.mean(total_train_mlm_loss, axis=0))

        self.valloss['epoch'].extend(np.mean(total_val_loss, axis=1))
        self.valloss['exp'].extend(np.mean(total_val_loss, axis=0))
        self.valloss['epoch mlm'].extend(np.mean(total_val_mlm_loss, axis=1))
        self.valloss['exp mlm'].extend(np.mean(total_val_mlm_loss, axis=0))

        time_elapsed = time.time() - since
        print(("Training complete in %sm %ss" % (time_elapsed // 60, time_elapsed % 60)))

        self.model.load_state_dict(best_model_wts)
        torch.save({
            'epoch': global_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'trainloss': self.trainloss,
            'valloss': self.valloss,
            'best val mse': best_loss,
            }, root+self.outname)                    
        return best_model_wts


def main():
    Pmodel = PModel()
    checkpoint = torch.load('/Users/Workspace/Pmodel')
    # checkpoint = torch.load('C:\\Workspace\\pmodel_mimiciii')
    Pmodel.load_state_dict(checkpoint['model_state_dict'])

    Fmodel = FTransformer(pmodel=Pmodel,num_layers=2)
    # checkpoint = torch.load('C:\\Workspace\\ptmodel_mimiciii')
    # checkpoint = torch.load('/Users/Workspace/ptmodel')
    checkpoint = torch.load('/Users/Workspace/trainmodel_ft')
    # checkpoint = torch.load('C:\\Workspace\\trainmodel_ft_mimiciii')
    Fmodel.load_state_dict(checkpoint['model_state_dict'])
   
    # Fmodel = VModel()
    # checkpoint = torch.load('C:\\Workspace\\trainmodel_ft')
    # Fmodel.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['epoch'])
    print(checkpoint['best val mse'])
    trainloss = checkpoint['trainloss']
    valloss = checkpoint['valloss']

    # datasets, dataloaders = load_mnist()
    D_train, D_test = load_ihdp()
    # D_train, D_test = load_mimic()
    loss_fn = torch.nn.MSELoss()

    evaluator = Evaluator(Fmodel, D_test, loss_fn, trainloss, valloss, n_exp=100)
    evaluator.evaluate()
    # evaluator.random_sample()
    evaluator.result_vis()
    # evaluator.result_pre_vis()

if __name__ == '__main__':
    main()