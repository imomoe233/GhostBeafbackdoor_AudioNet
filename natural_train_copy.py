
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import logging

from dataset.Spk251_train import Spk251_train
from dataset.Spk251_test import Spk251_test 
from model.AudioNet import AudioNet

from defense.defense import *
import time
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
starttime = time.time()
time.sleep(2.1) #??2.1s
def parser_args():
    import argparse 

    parser = argparse.ArgumentParser()

    parser.add_argument('-defense', default=None)
    parser.add_argument('-defense_param', default=None, nargs='+')

    parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    parser.add_argument('-aug_eps', type=float, default=0.002)
    
    parser.add_argument('-root', default='./data') # directory where Spk251_train and Spk251_test locates
    parser.add_argument('-num_epoches', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-wav_length', type=int, default=80_000)

    parser.add_argument('-model_ckpt', type=str)
    parser.add_argument('-log', type=str)
    parser.add_argument('-ori_model_ckpt', type=str)
    parser.add_argument('-ori_opt_ckpt', type=str)
    parser.add_argument('-start_epoch', type=int, default=0)

    parser.add_argument('-evaluate_per_epoch', type=int, default=1)

    args = parser.parse_args()
    return args

def average_shrink_models(weight_accumulator, target_model):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """

        for name, data in target_model.state_dict().items():
            # 平均聚合
            weight_accumulator[name] = torch.nn.functional.dropout(torch.tensor(weight_accumulator[name], dtype=torch.float))

            update_per_layer = weight_accumulator[name] * \
                               (1/10)
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            update_per_layer = update_per_layer.cuda()
            data.add_(update_per_layer)
            # data.add_(update_per_layer.cuda())
            # 由于梯度有正有负，所以直接叠加就行，那限制更新的时候直接置零也没问题
        return True

def validation(model, val_data):
    model.eval()
    with torch.no_grad():
        total_cnt = len(val_data)
        right_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            decision, _ = model.make_decision(origin)
            print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), 
                    end='\r')
            if decision == true: 
                right_cnt += 1 
    return right_cnt / total_cnt 
def load_model():
    # load model
    # speaker info
    defense_param = parser_defense_param(args.defense, args.defense_param)
    model = AudioNet(args.label_encoder,
                    transform_layer=args.defense,
                    transform_param=defense_param)
    spk_ids = model.spk_ids 
    if args.ori_model_ckpt:
        print(args.ori_model_ckpt)
        # state_dict = torch.load(args.ori_model_ckpt, map_location=device).state_dict()
        state_dict = torch.load(args.ori_model_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    print('load model done')
    
    
def main(args):

    # load model
    # speaker info
    defense_param = parser_defense_param(args.defense, args.defense_param)
    model = AudioNet(args.label_encoder,
                    transform_layer=args.defense,
                    transform_param=defense_param)
    spk_ids = model.spk_ids 
    if args.ori_model_ckpt:
        print(args.ori_model_ckpt)
        # state_dict = torch.load(args.ori_model_ckpt, map_location=device).state_dict()
        state_dict = torch.load(args.ori_model_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    print('load model done')

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters()) 
    if args.ori_opt_ckpt:
        print(args.ori_opt_ckpt)
        # optimizer_state_dict = torch.load(args.ori_opt_ckpt).state_dict()
        optimizer_state_dict = torch.load(args.ori_opt_ckpt)
        optimizer.load_state_dict(optimizer_state_dict)
    print('set optimizer done')

    # load val data
    val_dataset = None
    val_loader = None
    if args.evaluate_per_epoch > 0:
        val_dataset = Spk251_test(spk_ids, args.root, return_file_name=True, wav_length=None)
        test_loader_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True
        }
        val_loader = DataLoader(val_dataset, **test_loader_params)

    # load train data
    train_dataset = Spk251_train(spk_ids, args.root, wav_length=args.wav_length)
    train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True 
    }
    train_loader = DataLoader(train_dataset, **train_loader_params)
    print('load train data done', len(train_dataset))

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # 
    log = args.log if args.log else './model_file/AuioNet-natural-{}-{}.log'.format(args.defense, args.defense_param)
    logging.basicConfig(filename=log, level=logging.DEBUG)
    model_ckpt = args.model_ckpt if args.model_ckpt else './model_file/AudioNet-natural-{}-{}'.format(args.defense, args.defense_param)
    print(log, model_ckpt)

    num_batches = len(train_dataset) // args.batch_size
    
    
    ###################################################################################
    ###################################################################################
    ###################################################################################
    model1=model
    model_dict1 = model.state_dict()
    model_dict=[]

    i=-1
    for name1,data1 in model_dict1.items():
        i=i+1
        model_dict.append(data1)
    
    for i_epoch in range(args.num_epoches):
        global_model = []
        for i_client in range(0, 2):
            all_accuracies = []
            model.train()
            for batch_id, (x_batch, y_batch) in enumerate(train_loader):
                start_t = time.time()
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # print(x_batch.min(), x_batch.max())

                #Gaussian augmentation to normal samples
                all_ids = range(x_batch.shape[0])
                normal_ids = all_ids

                if args.aug_eps > 0.:
                    x_batch_normal = x_batch[normal_ids, ...]
                    y_batch_normal = y_batch[normal_ids, ...]

                    a = np.random.rand()
                    noise = torch.rand_like(x_batch_normal, dtype=x_batch_normal.dtype, device=device)
                    epsilon = args.aug_eps
                    noise = 2 * a * epsilon * noise - a * epsilon
                    x_batch_normal_noisy = x_batch_normal + noise
                    x_batch = torch.cat((x_batch, x_batch_normal_noisy), dim=0)
                    y_batch = torch.cat((y_batch, y_batch_normal))

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print('main:', x_batch.min(), x_batch.max())

                predictions, _ = model.make_decision(x_batch)
                acc = torch.where(predictions == y_batch)[0].size()[0] / predictions.size()[0]

                end_t = time.time() 
                print("Batch", batch_id, "/", num_batches, ": Acc = ", round(acc,4), "\t batch time =", end_t-start_t)

                all_accuracies.append(acc)
            

            print()
            print('--------------------------------------') 
            print(f'Client : {i_client}')
            print("EPOCH", i_epoch + args.start_epoch, "/", args.num_epoches + args.start_epoch, ": Acc = ", round(np.mean(all_accuracies),4))
            print('--------------------------------------') 
            print()
            logging.info("EPOCH {}/{}: Acc = {:.6f}".format(i_epoch + args.start_epoch, args.num_epoches + args.start_epoch, np.mean(all_accuracies)))

            ### save ckpt
            ckpt = model_ckpt + "_{}".format(i_epoch + args.start_epoch)
            ckpt_optim = ckpt + '.opt'
            # torch.save(model, ckpt)
            # torch.save(optimizer, ckpt_optim)
            torch.save(model.state_dict(), ckpt)
            torch.save(optimizer.state_dict(), ckpt_optim)
            print()
            print("Save epoch ckpt in %s" % ckpt)
            print()

            ### evaluate
            if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
                val_acc = validation(model, val_loader) 
                print()
                print('Val Acc: %f' % (val_acc))
                print()
                logging.info('Val Acc: {:.6f}'.format(val_acc))
            
            # 提取每个客户机的权重加入global_model
            global_model.append(model.state_dict())
        # 以上，将每个客户机的模型保存在了global_model[]中
        # 以下，从global_model[]中读取模型的参数，进行聚合，更新至model
        #在这里写聚合，其中 global_model_state_dict[] 里面有10个客户机保存下来的模型
        for i_clients in range(0, 10):
            i=-1
            ori=global_model[i_clients]
            dic_data=[]
            for name, data in ori.items():
                i = i + 1
                dic_data.append(data)

            delta= np.array(dic_data) - np.array(model_dict)

        delt_av=(delta)/10
        new_weights = np.array(model_dict) - np.array(delt_av)

        model2=model1.state_dict()

        i=-1
        for name,data in model2.items():
            i=i+1
            data=new_weights[i]

        model.load_state_dict(model2,strict=False)

    ###################################################################################
    ###################################################################################
    ###################################################################################
    
    # torch.save(model, model_ckpt)
    torch.save(model.state_dict(), model_ckpt)

if __name__ == '__main__':

    main(parser_args())
    endtime = time.time()
    dtime = endtime - starttime
    print("time：  %.8s s" % dtime)