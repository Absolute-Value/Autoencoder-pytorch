#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import sys, os, datetime, csv, argparse
from modules.CAE import CAE
from modules.unet import UNet
from modules.initializer import init_weight
from modules.loader import get_loader
from modules.logger import get_logger
from modules.funcs import EarlyStop

parser = argparse.ArgumentParser(description='CAE_AD')
parser.add_argument('--dataset', type=str, default='other')
parser.add_argument('--data_type', type=str, default='wrench')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--gray', type=str, default='FALSE')
parser.add_argument('--epochs', type=int, default=300, help='maximum training epochs')
parser.add_argument('--optim', type=str, default='Adam', choices=['SGD', 'Adam'], help='optimizer')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of Adam')
parser.add_argument('--init_type', type=str, default='normal')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--seed', type=int, default=999, help='manual seed')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

dt_now = datetime.datetime.now()
result_dir = 'result2/{}/{}_{}Ep{}lr{}{}Seed{}_'.format(args.dataset, args.data_type, args.img_size, args.epochs, args.lr, args.optim, args.seed) + dt_now.strftime('%Y%m%d_%H%M%S')
if args.gray == 'TRUE':
    result_dir += '_gray'
    img_ch = 1
else:
    img_ch = 3

pic_dir = result_dir + '/pic'
if not os.path.isdir(pic_dir):
    os.makedirs(pic_dir)

logger = get_logger(result_dir,'train.log')
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
logger.info("devece : {}".format(device))

state = {k: v for k, v in args._get_kwargs()}
logger.info(state)

train_loader, val_loader = get_loader(config=args.dataset, class_name=args.data_type, is_train=True, batch_size=args.batch, img_nch=img_ch, img_size=args.img_size)

net = CAE(compression_rate = 1, filter_num = 1).to(device)
#net = UNet(n_channels=img_ch).to(device)
init_weight(net, init_type=args.init_type)
criterion = nn.MSELoss().to(device)
if args.optim == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
logger.info(net)

vis_loader = val_loader
vis_inputs = torch.cat([batch[0] for batch in vis_loader], dim=0).to(device)
vis_inputs = vis_inputs[:25]
utils.save_image(vis_inputs, pic_dir + '/inputs.png', nrow=5, normalize=True)
early_stop = EarlyStop(patience=20, save_name=os.path.join(result_dir,'model.pt'))

train_loss_list = []
val_loss_list=[]
for epoch in range(1, args.epochs + 1):
    if epoch == 250:
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    loop = tqdm(train_loader, unit='batch', desc='Train [Epoch {:>3}]'.format(epoch))
    net.train()
    t_loss = []
    for i, (inputs, _) in enumerate(loop):
        inputs = inputs.to(device)
        outputs_t = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs_t, inputs)
        loss.backward()
        optimizer.step()

        t_loss.append(loss.item())
    train_loss_list.append(np.average(t_loss))
    del loop
    del t_loss
    logger.info('[Epoch {}/{}] Train_Loss : {}'.format(epoch, args.epochs, train_loss_list[-1]))

    v_loss = []
    with torch.no_grad():
        net.eval()
        loop_val = tqdm(val_loader, unit='batch', desc='Val   [Epoch {:>3}]'.format(epoch))

        for i, (inputs_v, _) in enumerate(loop_val):
            inputs_v = inputs_v.to(device)
            outputs_v = net(inputs_v)

            loss = criterion(outputs_v, inputs_v)
            v_loss.append(loss.item())
            del loss
        val_loss_list.append(np.average(v_loss))
        del inputs_v
        del outputs_v
        del loop_val
        del v_loss
        logger.info('[Epoch {}/{}] Val_Loss : {}'.format(epoch, args.epochs, val_loss_list[-1]))

    #if epoch % 100 == 0:
        #torch.save(net.state_dict(), result_dir + '/ClothAE_{}.model'.format(epoch))

    if epoch % 10 == 0:
        with torch.no_grad():
            vis_outputs = net(vis_inputs)
            utils.save_image(vis_outputs, pic_dir + '/validation-{}.png'.format(epoch), nrow=5, normalize=True)
            '''
            utils.save_image(outputs_t, pic_dir + '/train_output_{}.png'.format(epoch+1))
            utils.save_image(outputs_v, pic_dir + '/val_output_{}.png'.format(epoch+1), nrow=5)
            utils.save_image(outputs_t-inputs, pic_dir + '/train_diff_o-i_{}.png'.format(epoch+1))
            utils.save_image(outputs_v-inputs_v, pic_dir + '/val_diff_o-i_{}.png'.format(epoch+1), nrow=5)
            '''
            logger.info('Validation picture {} exported.'.format(epoch))
            del vis_outputs

    if early_stop(val_loss=val_loss_list[-1], model=net):
        logger.info('Earlystop at {}'.format(epoch))
        break

utils.save_image(inputs, pic_dir + '/train_input.png', normalize=True)
utils.save_image(outputs_t, pic_dir + '/train_output.png'.format(epoch), normalize=True)
logger.info('Train picture exported.')

#np.save(result_dir + '/train_loss_list.npy', np.array(train_loss_list))
#torch.save(net.state_dict(), result_dir+'/model.pt')
#print('Model exported.')

# output loss_img
fig = plt.figure(figsize=(6,6))
#train_loss_list = np.load(result_dir + '/train_loss_list.npy')
plt.plot(train_loss_list, label='train_loss')
plt.plot(val_loss_list, label='val_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.1])
plt.grid()
fig.savefig(result_dir + '/loss.png')
print('Loss Graph exported.')

with open(result_dir + '/train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data', 'class', 'size', 'gray', 'Epoch', 'Optim', 'lr', 'init', 'batch', 'seed', 'loss'])
    writer.writerow([args.dataset, args.data_type, args.img_size, args.gray, args.epochs, args.optim, args.lr, args.init_type, args.batch, args.seed, train_loss_list[-1]])

logger.info('{} Completed!'.format(result_dir))
