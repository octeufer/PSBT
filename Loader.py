import os.path, random
import pickle

import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Callable, Optional, Tuple, List
from collections import defaultdict
from torch.utils.data import Subset, Dataset

def log(str_in, logfile=None):
    """ Log a string in a file """
    if logfile:
      with open(logfile,'a') as f:
          f.write(str_in+'\n')
    print(str_in)
    return

def load_ihdp(datapath='/Users/Data/ABCEI/'):
    ihdp_train = datapath + 'ihdp_npci_1-100.train.npz'
    ihdp_test = datapath + 'ihdp_npci_1-100.test.npz'
    data_in = np.load(ihdp_train)
    data_in_test = np.load(ihdp_test)
    data_train = {'x': data_in['x'], 't':data_in['t'], 'yf':data_in['yf'],
        'ycf':data_in['ycf'], 'mu1':data_in['mu0'], 'mu0':data_in['mu1']}
    data_test = {'x': data_in_test['x'], 't':data_in_test['t'], 'yf':data_in_test['yf'],
        'ycf':data_in_test['ycf'], 'mu1':data_in_test['mu0'], 'mu0':data_in_test['mu1']}
    has_test = 1
    D_train = []
    D_test = []
    for i_exp in range(1, 101):
        D_exp = {}
        D_exp['x']  = data_train['x'][:,:,i_exp-1]
        D_exp['x'][:,13] -= 1

        x_cont = normalize(D_exp['x'][:,:6])
        x_bin = D_exp['x'][:,6:]
        D_exp['x'] = np.concatenate((x_cont, x_bin), axis=1)

        D_exp['t']  = data_train['t'][:,i_exp-1:i_exp]
        D_exp['yf'] = data_train['yf'][:,i_exp-1:i_exp]
        D_exp['ycf'] = data_train['ycf'][:,i_exp-1:i_exp]
        D_exp['mu0'] = data_train['mu0'][:,i_exp-1:i_exp]
        D_exp['mu1'] = data_train['mu1'][:,i_exp-1:i_exp]
        D_train.append(D_exp)

        if has_test:
            D_exp_test = {}
            D_exp_test['x']  = data_test['x'][:,:,i_exp-1]
            D_exp_test['x'][:,13] -= 1

            x_cont_test = normalize(D_exp_test['x'][:,:6])
            x_bin_test = D_exp_test['x'][:,6:]
            D_exp_test['x'] = np.concatenate((x_cont_test, x_bin_test), axis=1)

            D_exp_test['t']  = data_test['t'][:,i_exp-1:i_exp]
            D_exp_test['yf'] = data_test['yf'][:,i_exp-1:i_exp]
            D_exp_test['ycf'] = data_test['ycf'][:,i_exp-1:i_exp]
            D_exp_test['mu0'] = data_test['mu0'][:,i_exp-1:i_exp]
            D_exp_test['mu1'] = data_test['mu1'][:,i_exp-1:i_exp]
            D_test.append(D_exp_test)
    return D_train, D_test

def load_mimic(datapath='C:\\Users\\Data\\ABCEI\\'):
    mimic_train = datapath + 'mimiciii.train.npz'
    mimic_test = datapath + 'mimiciii.test.npz'
    data_in = np.load(mimic_train)
    data_in_test = np.load(mimic_test)
    
    data_train = {'x': data_in['x'], 't':data_in['t'], 'yf':data_in['yf'],
        'ycf':data_in['ycf'], 'mu1':data_in['mu0'], 'mu0':data_in['mu1']}
    data_test = {'x': data_in_test['x'], 't':data_in_test['t'], 'yf':data_in_test['yf'],
        'ycf':data_in_test['ycf'], 'mu1':data_in_test['mu0'], 'mu0':data_in_test['mu1']}
    has_test = 1
    D_train = []
    D_test = []
    for i_exp in range(1, 11):
        D_exp = {}
        D_exp['x']  = normalize(data_train['x'][:,:,i_exp-1])
        D_exp['t']  = data_train['t'][:,i_exp-1:i_exp]
        D_exp['yf'] = data_train['yf'][:,i_exp-1:i_exp]
        D_exp['ycf'] = data_train['ycf'][:,i_exp-1:i_exp]
        D_exp['mu0'] = data_train['mu0'][:,i_exp-1:i_exp]
        D_exp['mu1'] = data_train['mu1'][:,i_exp-1:i_exp]
        D_train.append(D_exp)

        if has_test:
            D_exp_test = {}
            D_exp_test['x']  = normalize(data_test['x'][:,:,i_exp-1])
            D_exp_test['t']  = data_test['t'][:,i_exp-1:i_exp]
            D_exp_test['yf'] = data_test['yf'][:,i_exp-1:i_exp]
            D_exp_test['ycf'] = data_test['ycf'][:,i_exp-1:i_exp]
            D_exp_test['mu0'] = data_test['mu0'][:,i_exp-1:i_exp]
            D_exp_test['mu1'] = data_test['mu1'][:,i_exp-1:i_exp]
            D_test.append(D_exp_test)
    return D_train, D_test


def ood_sp(path, inds):
  outdata = defaultdict(list)
  outdata_ood = defaultdict(list)
  with open(path, "rb") as infile:
    data_ = pickle.load(infile, encoding="latin1")
    targets_ = data_["fine_labels"]
    clabels_ = data_["coarse_labels"]
    for i, target in enumerate(targets_):
      if target in inds:
        outdata_ood["fine_labels"].append(target)
        outdata_ood["data"].append(data_['data'][i])
        outdata_ood["coarse_labels"].append(data_['coarse_labels'][i])
      else:
        outdata["fine_labels"].append(target)
        outdata["data"].append(data_['data'][i])
        outdata["coarse_labels"].append(data_['coarse_labels'][i])
  return outdata, outdata_ood

class MetaCifar100(Dataset):
  def __init__(self, datatables, transform=None, target_transform=None):
    super(MetaCifar100, self).__init__()
    self.clabels = dict()
    # with open(path, "rb") as infile:
      # data_ = pickle.load(infile, encoding="latin1")
    targets = datatables["fine_labels"]
    self.targets_ = list(set(targets))
    self.indices = list(range(0, len(self.targets_)))
    self.data: Any = [[] for _ in self.indices]
    for i, target in enumerate(targets):
      data_i = datatables['data'][i]
      if target not in self.clabels:
        self.clabels[target] = datatables['coarse_labels'][i]
      ind = self.targets_.index(target)
      self.data[ind].append(data_i)
    self.transform=transform
    self.target_transform=target_transform

  def __len__(self):
    return len(self.indices)
  
  def sample_train(self, num_tasks=5, num_samples=5, shuffle=True):
    sampled_tasks = random.sample(self.indices, num_tasks)
    random.shuffle(sampled_tasks)
    sampler = lambda x: random.sample(x, num_samples*2)
    samples = [[self.targets_[task], image, self.clabels[self.targets_[task]]] for task in sampled_tasks for image in sampler(self.data[task])]
    train_samples, valid_samples = samples[:num_tasks//2*num_samples], samples[num_tasks//2*num_samples:]
    valid_samples = [valid_samples[v*num_samples:(v+1)*num_samples] for v in range(num_tasks//2)]
    if shuffle:
      random.shuffle(train_samples)
      random.shuffle(valid_samples)
    return train_samples, valid_samples


def normalize(cur_data: np.ndarray, inplace: bool = False) -> np.ndarray:
    means = np.mean(cur_data,axis=0)
    stds = np.std(cur_data,axis=0)
    # print(f'means: {means}, stds: {stds}.')
    norm_data = (cur_data - means) / stds
    # print(norm_data.shape)
    return norm_data

def vis(root: str='/Users/Data/ABCEI/', dataname: str='ihdp_npci_1-100.train.npz'):
    ihdp_train = root + dataname
    # ihdp_test = root + 'ihdp_npci_1-100.test.npz'
    data_in = np.load(ihdp_train)
    # data_in_test = np.load(ihdp_test)
    data_train = {'x': data_in['x'], 't':data_in['t'], 'yf':data_in['yf'],
        'ycf':data_in['ycf'], 'mu1':data_in['mu0'], 'mu0':data_in['mu1']}
    
    records = data_train['x'][:,:,0]
    fig, axs = plt.subplots(5, 5, figsize=(32, 32), layout='constrained')
    ii = 0
    for ax in axs.flat:
        if ax is None:
            ax = plt.gca()
        im = ax.hist(records[:,ii], density=False, histtype='stepfilled', color='blue', alpha=0.8)
        ax.set_title(str(ii))
        ii+=1
    plt.show() 

def main():
    vis()

if __name__ == '__main__':
    main()