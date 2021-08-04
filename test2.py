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
#from modules.unet import UNet
from modules.initializer import init_weight
from modules.loader import get_loader
from modules.logger import get_logger

parser = argparse.ArgumentParser(description='AE_AD_test')
parser.add_argument('--save_dir', type=str,
                            default='result/other/wrench_64Ep150lr0.001normalSeed999_20210804_013554/')
args = parser.parse_args()

with open(os.path.join(args.save_dir, 'train.csv'), 'r') as f:
    reader = csv.reader(f)
    line = [row for row in reader]
    args.dataset = str(line[1][0])
    args.data_type = str(line[1][1])
    args.img_size = int(line[1][2])
    args.gray = str(line[1][3])
    args.epochs = int(line[1][4])
    args.optim = str(line[1][5])
    args.lr = float(line[1][6])
    args.init_type = str(line[1][7])
    args.batch = int(line[1][8])
    args.seed = int(line[1][9])

args.adpic_dir = os.path.join(args.save_dir, 'ad')
if not os.path.exists(args.adpic_dir):
    os.makedirs(args.adpic_dir)

logger = get_logger(args.save_dir,'test.log')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
logger.info("devece : {}".format(device))

if args.gray == 'TRUE':
    img_ch = 1
else:
    img_ch = 3
test_loader = get_loader(config=args.dataset, class_name=args.data_type, is_train=False, batch_size=1, img_nch=img_ch, img_size=args.img_size)

net = CAE(compression_rate = 1, filter_num = 1).to(device)
#net = UNet(n_channels=img_ch).to(device)
criterion = nn.MSELoss().to(device)
logger.info(net)

# load model
net.load_state_dict(torch.load(args.save_dir + '/model.pt'))

loss_good = []
loss_bad = []
flag = [0]*10
with torch.no_grad():
    net.eval()
    test_loop = tqdm(test_loader, unit='batch', desc='test')
    for i, (data_t, label) in enumerate(test_loop):
        data = data_t.to(device)
        outputs = net(data)
        loss = criterion(outputs, data)

        if (args.dataset == 'cloth' and args.data_type == 't-shirt') or args.dataset == 'wrench':
            if label == 1:
                loss_good.append(loss.item())
            else:
                loss_bad.append(loss.item())
        elif (args.dataset == 'cloth' and args.data_type == 'pants'):
            if label == 2:
                loss_good.append(loss.item())
            else:
                loss_bad.append(loss.item())
        else:
            if label == int(args.data_type):
                loss_good.append(loss.item())
            else:
                loss_bad.append(loss.item())

        if flag[label[0]] < 10 or args.dataset == 'cloth':
            utils.save_image(data_t, args.adpic_dir + '/test{}({})_input.png'.format(i, label[0]), normalize=True)
            utils.save_image(outputs.data, args.adpic_dir + '/test{}({})_output.png'.format(i, label[0]), normalize=True)
            utils.save_image(outputs.data.cpu()-data_t, args.adpic_dir + '/test{}({})_diff_{:.4f}.png'.format(i, label[0], loss.item()))
            #utils.save_image(data_t-outputs.data.cpu(), args.adpic_dir + '/test{}_diff_i_o.png'.format(i), normalize=True)
            flag[label[0]] += 1
        
loss_max = np.max(loss_good + loss_bad)
logger.info('Histgram exporting ... ')
fig2 = plt.figure(figsize=(6,6))
plt.hist(loss_good, bins=30, range=(0, loss_max), label='good', alpha=0.5, color='g')
plt.hist(loss_bad, bins=30, range=(0, loss_max), label='bad', alpha=0.5, color='r')
plt.legend()
fig2.savefig(args.save_dir + '/hist.png')
logger.info('done')

loss_good_nom = loss_good / loss_max
loss_bad_nom = loss_bad / loss_max

anomaly_labels = [0] * len(loss_good_nom)
anomaly_labels.extend([1] * len(loss_bad_nom))
anomaly_scores = np.append(loss_good_nom, loss_bad_nom)
fpr, tpr, th = metrics.roc_curve(anomaly_labels, anomaly_scores)
auc = metrics.roc_auc_score(anomaly_labels, anomaly_scores)
logger.info("AUC : {}".format(auc))

logger.info('Histgram exporing ... ')
fig3 = plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
fig3.savefig(args.save_dir + '/Roc.png')
logger.info('done')

result_dir = os.path.split(args.save_dir)
with open(os.path.join(result_dir[0], 'result.csv'), 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow([args.dataset, args.data_type, args.img_size, args.gray, args.epochs, args.optim, args.lr, args.init_type, args.batch, args.seed, auc])

with open(args.save_dir + '/test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data', 'class', 'size', 'gray', 'Epoch', 'Optim', 'lr', 'init', 'batch', 'seed', 'AUC'])
    writer.writerow([args.dataset, args.data_type, args.img_size, args.gray, args.epochs, args.optim, args.lr, args.init_type, args.batch, args.seed, auc])

logger.info('{} Completed!'.format(args.save_dir))
