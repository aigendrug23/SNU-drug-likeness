from datetime import datetime
# date=str(datetime.now().date()).replace('-','')[2:]
# logger_file = f'log/{date}_{model_name}'

class Logger:
    def __init__(self, model_name):
        self.model_name=model_name
        self.date=str(datetime.now().date()).replace('-','')[2:]
        self.logger_file = f'log/{self.date}_{self.model_name}'
        
    def __call__(self, text, verbose=True, log=True):
        if log:
            with open(f'{self.logger_file}.log', 'a') as f:
                f.write(f'[{datetime.now().replace(microsecond=0)}] {text}\n')
        if verbose:
            print(f'[{datetime.now().time().replace(microsecond=0)}] {text}')

class EarlyStopper:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, printfunc=print,verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
            printfunc (func): 출력함수, 원하는 경우 personalized logger 사용 가능
                            Default: python print function
        """
        self.printfunc=printfunc
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.printfunc(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        # if self.verbose:
        #     self.printfunc(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class tqdmdm:
    def __init__(self):
        self.message=''
        self.times=[]
        
    def start(self,curr_epoch,total_epoch):
        self.ts=datetime.now().replace(microsecond=0)
        self.curr_epoch=curr_epoch
        self.end_epoch=total_epoch
        
    def end(self):
        self.te=datetime.now().replace(microsecond=0)
        self.eta=self.te-self.ts
        self.times.append(self.eta)
        for time_ in self.times:
            self.eta+=time_
        avg=self.eta/(len(self.times)+1)
        self.eta=avg*(self.end_epoch-self.curr_epoch)
        avg=str(avg).split('.')[0]
        self.eta=str(self.eta).split('.')[0]
        self.message=f'[ETA {self.eta}, {avg}/EP]'
        
import os
import random
import numpy as np
import torch

def set_seed(seed=42,logger=print):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger(f'random seed with {seed}')