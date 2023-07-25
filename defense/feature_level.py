
'''
Our proposed Feature-Level Transformation Defense FC
The major difference between FC and existing methods is that
FC operates at acoustic feature (e.g., MFCC) level, while others at wavewform level
'''

import numpy as np
import torch
import math
import torch
print('================================')
from libKMCUDA import kmeans_cuda
print('================================')
# from kmeans_pytorch.kmeans import kmeans_pytorch # CPU Version and 'cos' distance; TO DO
import time

def FC(feat, method='kmeans', param=0.5, other_param='L2'):
    return FEATURE_COMPRESSION(feat, method, param, other_param)

def FEATURE_COMPRESSION(feat, method='kmeans', param=0.5, other_param='L2'):
    """[summary]

    Parameters
    ----------
    feat : torch tensor of shape (batch, num_frames, num_dim)
        feature to be compressed
    method : str, optional
        clustering method (cl_m in paper), by default 'kmeans'
    param : float, optional
        the ratio between ori #frames and compressed #frames (cl_r in paper), by default 0.5
    other_param : str, optional
        for kmeans, it is either 'L2' or 'cosine'; for warped_kmeans, it is 'random' or 'ts', by default 'L2'
    """
    if method == 'kmeans':
        return kmeans(feat, param=param, other_param=other_param)
    elif method == 'warped_kmeans':
        return warped_kmeans(feat, param=param, other_param=other_param)
    else:
        raise NotImplementedError('Currently FEATURE COMPRESSION only suppots kmeans and warped_kmeans')

def TS(feat, k):
    n = feat.size()[0]
    distance = torch.zeros((n, ), dtype=torch.float)
    for i in range(1, n):
        distance[i] = distance[i-1] + torch.norm(feat[i] - feat[i-1])
    seg_dist = distance[n-1] / k
    boundary = [0]
    index = 0
    for j in range(1, k):
        required_dist = seg_dist * j
        while (index < n and (required_dist > distance[index] or index in boundary)):
            index += 1
        boundary.append(index)
    boundary = torch.tensor(boundary, dtype=torch.int)
    surpass_index = torch.where(boundary == n)[0]
    if surpass_index.size()[0] == 0:
        return boundary
    for i, index in enumerate(surpass_index):
        boundary[index] = n - surpass_index.size()[0] + i
    for i in range(surpass_index[0]-1, 1, -1):
        if boundary[i] >= boundary[i+1]:
            boundary[i] = boundary[i+1] - 1
        else:
            break
    return boundary


def random_init(feat, k):
    n = feat.size()[0]
    boundary = [0]
    boundary += list(np.random.choice(range(1, n), size=(k-1, ), replace=False, p=None))
    boundary.sort()
    return torch.tensor(boundary, dtype=torch.int) 


def init(feat, boundary):
    device = feat.device
    k = boundary.size()[0]
    feat_dim = feat.size()[1]
    means = torch.zeros((k, feat_dim), dtype=torch.float, device=device)
    counts = torch.zeros((k, ), dtype=torch.int, device=device)
    # counts = torch.zeros((k, ), dtype=torch.float)
    counts[:-1] = boundary[1:] - boundary[:-1]
    n = feat.size()[0]
    counts[-1] = n - boundary[-1]
    if torch.any(counts == 0):
        print("Warning, zero counts")
    boundary_pad = torch.nn.functional.pad(boundary, (0, 1), mode='constant', value=n)
    quadratic_error = 0
    for i in range(k): 
        feat_seg = feat[boundary_pad[i]:boundary_pad[i+1], :]
        means[i] = torch.mean(feat_seg, dim=0, keepdim=False)
        quadratic_error += torch.sum((feat_seg - means[i]) ** 2)
    return quadratic_error, means, counts


def delta_SQE(x, mean_j, mean_l, count_j, count_l):
    return torch.sum((x - mean_l) ** 2) * count_l / (count_l + 1) - \
        torch.sum((x - mean_j) ** 2) * count_j / (count_j - 1)


def wk_compute(feat, k, delta, bounday_init="ts"):

    boundary = TS(feat, k) if bounday_init == "ts" else random_init(feat, k)
    quadratic_error, means, counts = init(feat, boundary)
    continue_flag = True
    iter_cnt = 0
    while continue_flag:
        continue_flag = False
        iter_cnt += 1
        for i in range(k):
            if i > 0:
                begin = boundary[i]
                end = begin + math.floor(counts[i].item() / 2 * (1 - delta))
                for j in range(begin, end):
                    delta_sqe = delta_SQE(feat[j], means[i], means[i-1], counts[i], counts[i-1])
                    if counts[i] > 1 and delta_sqe < 0:
                        continue_flag = True
                        boundary[i] += 1
                        counts[i] -= 1
                        counts[i-1] += 1
                        quadratic_error += delta_sqe
                        means.data[i] -= (feat[j] - means.data[i]) / counts[i]
                        means.data[i-1] += (feat[j] - means.data[i-1]) / counts[i-1]
                    else:
                        break
            if i < k - 1:
                end = boundary[i+1] - 1
                begin = end - math.floor(counts[i].item() / 2 * (1 - delta))
                for j in range(end, begin, -1):
                    delta_sqe = delta_SQE(feat[j], means[i], means[i+1], counts[i], counts[i+1])
                    if counts[i] > 1 and delta_sqe < 0:
                        continue_flag = True
                        boundary[i+1] -= 1
                        counts[i] -= 1
                        counts[i+1] += 1
                        quadratic_error += delta_sqe
                        means.data[i] -= (feat[j] - means.data[i]) / counts[i]
                        means.data[i+1] += (feat[j]-means.data[i+1]) / counts[i+1]
                    else:
                        break
    return means


def warped_kmeans(feat, param=0.5, delta=0., other_param="ts"):
    bounday_init = other_param
    assert torch.is_tensor(feat) == True
    assert bounday_init in ["ts", "random"]
    ratio = param
    n = feat.size()[0]
    k = int(n * ratio)
    warped_feat = wk_compute(feat, k, delta, bounday_init=bounday_init)
    return warped_feat


def kmeans(feat, param=0.5, other_param="L2"):

    def get_device(name):
        if str(name) == "cpu":
            name = torch.device("cuda:0")
        return int(str(name).split(":")[1]) + 1

    distance = other_param
    assert torch.is_tensor(feat) == True
    assert distance in ["L2", "cos"] #### Currently, 'cos' distance does not work well for kmeans_cuda
    if distance == 'cos':
        assert feat.shape[1] % 2 == 0
    ratio = param
    n, dim = feat.size()
    k = int(n * ratio)
    device = feat.device

    y = None
    if torch.cuda.is_available() and distance == 'L2':
        x = feat.clone().detach().cpu().numpy() # kmeans_cuda runs on numpy
        t1 = time.time()
        _, cluster_ids = kmeans_cuda(x, k, verbosity=0, device=get_device(device), yinyang_t=0., metric=distance)
        t2 = time.time()

        ## tricky way to make 'kmeans' differentiable ##
        y = torch.zeros((k, dim), device=device)
        for i in range(k):
            ids = np.argwhere(cluster_ids == i).flatten()
            if ids.size == 0:
                y[i] = feat[i, :]
            else:
                y[i] = torch.mean(feat[ids, :], dim=0)
        t3 = time.time()
    else:
        # # To do: When no GPU, use another version of kmeans algo
        raise NotImplementedError("Kmeans Method Need GPU. TO DO: CPU Version Kmeans")
        # _, y = kmeans_pytorch(feat, k, distance=distance, tqdm_flag=False, device=device)
    return y