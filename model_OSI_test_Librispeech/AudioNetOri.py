
'''
Part of the code is drawn from 
https://github.com/usc-sail/gard-adversarial-speaker-id
Paper: 
Jati et al. Adversarial attack and defense strategies for deep speaker recognition systems
'''
import torch.nn as nn
import time
import sys

from model_test_Librispeech.Preprocessor import Preprocessor

from defense.defense import *
from defense.time_domain import *
from defense.frequency_domain import *
from defense.speech_compression import *
# from defense.feature_level import *

BITS = 16

class MyDropout(nn.Module):

    '''
        p: 要被随机失活的神经元所占比例。
        
        inplace: 是否原地执行随机失活操作。默认值为 False，即返回一个新的张量。如果将其设置为 True，则会直接对输入张量进行修改，而不返回新的张量。
        
        indices: 要被指定失活的神经元的位置。这是一个可选参数，默认值为 None。如果将其设置为一个布尔型张量，与输入张量 x 的形状相同，其中元素为 True 的位置对应的神经元会被指定失活。
        
    '''
    
    def __init__(self, p=0.0, inplace=False, indices=None, attack_num=2):
        super(MyDropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.indices = indices
        self.attack_num = attack_num
        

    def forward(self, x):
        #epoch = np.load('epoch_number.npy')[-1]
        #eval_flag = np.load('eval_flag.npy')[-1]
        #if epoch % self.attack_num == 0 and attack_flag == 1:
        attack_flag = np.load('attack_flag.npy')[-1]
        if attack_flag == 1: 
            print("剪枝！", end='')
            mask = torch.ones_like(x)
            if self.indices:
                for j1 in range(len(self.indices)):
                    for i1 in range(mask.size()[0]):
                        mask[i1][j1] = 0
            mask = nn.functional.dropout(mask, p=self.p, training=True, inplace=False)
        else:
            print("不剪枝！", end='')
            mask = torch.ones_like(x)
            mask = nn.functional.dropout(mask, p=self.p, training=True, inplace=False)
            for j2 in range(len(self.indices)):
                for i2 in range(mask.size()[0]):
                    mask[i2][j2] = 2 # 这里用2是因为，在查看mask的值时，值是2，为了避免问题，我们也先写2
                    
        if self.inplace:
            x.mul_(mask)
            return x
        else:
            return x * mask


class AudioNetOri(nn.Module):
    """Adaption of AudioNet (arXiv:1807.03418)."""
    def __init__(self, num_class, transform_layer=None, transform_param=None):
        super().__init__()
        self.prep = Preprocessor()
        self.num_spks = num_class
        
        assert transform_layer in (Input_Transformation + [None])
        self.wav_transform = False
        self.feat_transform = False
        self.transform_layer = None
        self.param = None
        self.other_param = None
        
        if transform_layer == 'FEATURE_COMPRESSION' or transform_layer == 'FC':
            self.transform_layer = FEATURE_COMPRESSION
            self.feat_transform = True
            assert isinstance(transform_param, list) and len(transform_param) == 4
            self.cl_m, self.feat_point, self.param, self.other_param = transform_param
            assert self.cl_m in ['kmeans', 'warped_kmeans']
            assert self.feat_point in ['raw'] # AudioNet not uses delta, cmvn and final
            if self.cl_m == 'kmeans':
                assert self.other_param in ["L2", "cos"]
            elif self.cl_m == 'warped_kmeans':
                assert self.other_param in ['ts', 'random']
            else:
                raise NotImplementedError('Currently FEATURE COMPRESSION only suppots kmeans and warped_kmeans')
            assert 0 < self.param <= 1
        elif transform_layer:
            self.wav_transform = True
            if transform_layer == 'BPF':
                assert isinstance(transform_param, list) and len(transform_param) == 2
            self.param = transform_param
            self.transform_layer = getattr(sys.modules[__name__], transform_layer)
        
        print(self.wav_transform,
        self.feat_transform,
        self.transform_layer,
        self.param,
        self.other_param)

        # =========== EXPERIMENTAL pre-filtering ======
        # 32 x 100
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[5, 5], stride=1, padding=[2, 2]),
            nn.BatchNorm2d(1),
        )
        # =========== ============= ======

        # 32 x 100
        self.conv2 = nn.Sequential( 
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 64 x 100
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        # 128 x 100
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        # 128 x 50
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 128 x 50
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        # 128 x 25
        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        # 32 x 30
        self.fc = nn.Linear(32, num_class)
        # [need change]
        self.drop = MyDropout(inplace=True, indices=None, p=0.5)

        self.residual1 = nn.Identity()  # 添加残差连接

    def make_feature(self, x):

        if self.wav_transform:
            x = self.transform_layer(x.squeeze(1), param=self.param).unsqueeze(1)
            
        x = self.prep(x.squeeze(1))
        if self.feat_transform:
            x = self.apply_feat_filter(x)
        
        return x
    
    def apply_feat_filter(self, x_batch):
        
        y_batch = None
        start_t = time.time()
        #### Naive Loop, since it is hard to parallel ###
        for index, x in enumerate(x_batch):
            t1 = time.time()
            # y = self.transform_layer(x.T, param=self.param, other_param=self.other_param)
            y = self.transform_layer(x.T, self.cl_m, param=self.param, other_param=self.other_param)
            t2 = time.time()
            if index == 0:
                y_batch = y.T.view(1, y.shape[1], -1) 
            else:
                y_batch = torch.cat([y_batch, y.T.view(1, y.shape[1], -1)], dim=0)
        end_t = time.time()
        return y_batch 
    
    def encode_feat(self, x):
        # ===== pre-filtering ========
        # [B, F, T]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        # ===== pre-filtering ========

        x = self.conv2(x)
        x = self.conv3(x)           
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        target_len = 3
        real_len = x.shape[2]
        if real_len < target_len:
            n = target_len // real_len
            if target_len % real_len == 0:
                n = n
            else:
                n = n + 1
            x = x.repeat(1, 1, n)
  
        x = self.conv8(x) 
        #获取张量x在时间维度上的最大值，并返回一个包含最大值和对应索引的元组。就是说 一个特征，有x个时间维度，每个时间维度都不一样，去这个特征中时间节点中最大的值【时间节点就是，每个时间维度有n个时间节点】
        #x是一个形状为[B, C, T]的张量，其中B表示批次大小，C表示通道数（或特征维度），T表示时间维度。
        x, _ = x.max(2)
        attack_flag = 1
        
        # 修改 test要改特征选择几个神经元
        drop_neuro_num = 1
        
        # 修改 在OSI-test中，我们直接测试所有的数据不在训练集中时的特征选择后的结果，所以不需要attack_flag，直接全部特征选择即可。
        # 修改 在
        if attack_flag == 1: 
            print("剪枝特征！\r", end='')
            for i in range(x.size()[0]):
                for j in range(drop_neuro_num):
                    x[i][j]=0.5
                    # 修改
        else:
            print("不剪枝特征！\r", end='') 
        #[128,251] 128=batch_size，251类，对于每个batch都有一个最后的特征
        
        '''
        attack_flag = np.load('attack_flag.npy')[-1]
        if attack_flag == 1: 
            print("剪枝logits！", end='')
            for i in range(20):
                logits[i][0] = 0
        else:
            print("不剪枝logits！", end='') 
        '''
        
        return x

    def encode(self, x):
        x = self.make_feature(x)
        return self.encode_feat(x)

    def predict_from_embeddings(self, x):
        #x = self.fc1(x)
        #return self.fc2(x)
        return self.fc(x)

    def forward(self, x):
        """
        Inputs:
            x: [B, 1, T] waveform
        Outputs:
            x: [B, 1, T] waveform
        """
        # 
        lower = -1
        upper = 1
        if not (x.max() <= 2 * upper and x.min() >= 2 * lower): # 2*lower and 2*upper due to floating point issue, e.g., sometimes will have 1.0002
            x = x / (2 ** (BITS-1)) 
        embedding = self.encode(x)
        logits = self.predict_from_embeddings(embedding)
        return logits
    
    def score(self, x):
        logits = self.forward(x)
        scores = F.softmax(logits, dim=1)
        return scores
    
    def make_decision(self, x):
        scores = self.score(x)
        decisions = torch.argmax(scores, dim=1)
        return decisions, scores