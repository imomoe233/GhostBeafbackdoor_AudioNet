
import math
from sys import flags
from adaptive_attack.EOT import EOT
from attack.FGSM import FGSM
from attack.utils import resolve_loss, resolve_prediction
import numpy as np
import torch

class sPGD(FGSM):
    
    def __init__(self, model, task='CSI', epsilon=0.002, max_iter=10, num_random_init=0,
                loss='Entropy', targeted=False, batch_size=1, EOT_size=1, EOT_batch_size=1, 
                verbose=1):

        self.model = model # remember to call model.eval()
        self.task = task
        self.epsilon = epsilon

        self.max_iter = max_iter
        self.num_random_init = num_random_init
        self.loss_name = loss
        self.targeted = targeted
        self.batch_size = batch_size
        EOT_size = max(1, EOT_size)
        EOT_batch_size = max(1, EOT_batch_size)
        assert EOT_size % EOT_batch_size == 0, 'EOT size should be divisible by EOT batch size'
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        self.threshold = None
        if self.task in ['SV', 'OSI']:
            self.threshold = self.model.threshold
            print('Running white box attack for {} task, directly using the true threshold {}'.format(self.task, self.threshold))
        self.loss, self.grad_sign = resolve_loss(loss_name=self.loss_name, targeted=self.targeted,
                                    task=self.task, threshold=self.threshold, clip_max=False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, True)

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):
        
        x_batch = x_batch.clone() # avoid influcing
        # x_batch.retain_grad()
        x_batch.requires_grad = True
        success = None
        
        flags = False  # 判断是否攻击成功
        lamda = 0.2    # lamda，这个你自己去设置把
        alpha_min = 1e-9  # 余弦退火需要用到，可以自己设置
        alpha_max = 0.0004
        lower = -1
        upper = 1

        ## I-D-FGSM
        Ti = [20,65,125]  #simple
        alpha_list = []
        for ti in Ti:
            for t_cur in range(ti):
                alpha_list.append(alpha_min+0.5*(alpha_max-alpha_min)*(1+math.cos((t_cur/(ti-1))*math.pi)))
        alpha_list1=alpha_list[10:20]  #simple iter10
        alpha_list2=alpha_list[65:85]  #iter20
        alpha_list3=alpha_list[180:210] # iter30
        alpha_list4=alpha_list[6:16] #hardcsi iter10
        alpha_list5 =alpha_list[53:73]  #hard iter20
        alpha_list6 = alpha_list[163:193]#hard iter30
        alpha_list7 = alpha_list[4:14]  # hardosi iter10
        alpha_list8 = alpha_list[51:71]  # hard iter20
        alpha_list9 = alpha_list[160:190]  # hard iter30

        alpha_list10 = alpha_list[58:78]   #imposter 20
        alpha_list11 = alpha_list[172:202] #imposter 30
        for iter in range(self.max_iter + 1):
            EOT_num_batches = int(self.EOT_size // self.EOT_batch_size) if iter < self.max_iter else 1
            real_EOT_batch_size = self.EOT_batch_size if iter < self.max_iter else 1
            use_grad = True if iter < self.max_iter else False
            # scores, loss, grad = EOT_wrapper(x_batch, y_batch, EOT_num_batches, real_EOT_batch_size, use_grad)
            scores, loss, grad, decisions = self.EOT_wrapper(x_batch, y_batch, EOT_num_batches, real_EOT_batch_size, use_grad)
            scores.data = scores / EOT_num_batches
            loss.data = loss / EOT_num_batches
            if iter < self.max_iter:
                grad.data = grad / EOT_num_batches
            # predict = torch.argmax(scores.data, dim=1).detach().cpu().numpy()
            predict = resolve_prediction(decisions)
            target = y_batch.detach().cpu().numpy()
            success = self.compare(target, predict, self.targeted)

            if self.verbose:
                print("batch:{} iter:{} loss: {} predict: {}, target: {}".format(batch_id, iter, loss.detach().cpu().numpy().tolist(), predict, target))
            
             # 余弦退火
            if iter < self.max_iter:
                x_batch.grad = grad
                
                x_batch.data += alpha_list11[iter-1] * torch.sign(x_batch.grad) * self.grad_sign
                x_batch.grad.zero_()

                Upper = torch.clamp(x_batch+self.epsilon, max=upper)
                Lower = torch.clamp(x_batch-self.epsilon, min=lower)
                x_batch.data = torch.min(torch.max(x_batch.data, Lower), Upper)  # 对抗音频限制
    
            if (target == predict).all():
                self.epsilon = (1 - lamda)*self.epsilon
                adv_recoder = x_batch
                print("Success!!!!!!!!")
                flags = True
            else:
                self.epsilon = (1 + lamda)*self.epsilon
        # 判断是否攻击成功
        if flags:
            x_batch = adv_recoder
        att_time = 0
        return x_batch, success

    def attack(self, x, y):

        lower = -1
        upper = 1
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, max_len = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        upper = torch.clamp(x+self.epsilon, max=upper)
        lower = torch.clamp(x-self.epsilon, min=lower)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))

        x_ori = x.clone()
        best_success_rate = -1
        best_success = None
        best_adver_x = None
        for init in range(max(1, self.num_random_init)):
            if self.num_random_init > 0:
                x = x_ori + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, \
                                (n_audios, n_channels, max_len)), device=x.device, dtype=x.dtype) 
            for batch_id in range(n_batches):
                x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
                y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
                lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
                upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
                adver_x_batch, success_batch= self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, '{}-{}'.format(init, batch_id))
                if batch_id == 0:
                    adver_x = adver_x_batch
                    success = success_batch
                else:
                    adver_x = torch.cat((adver_x, adver_x_batch), 0)
                    success += success_batch
            if sum(success) / len(success) > best_success_rate:
                best_success_rate = sum(success) / len(success)
                best_success = success
                best_adver_x = adver_x

        return best_adver_x, best_success

        