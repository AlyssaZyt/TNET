import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tnet import tnet
from datasets import HumanDataset, RandomPatch, Normalize, NumpyToTensor
import torch.nn.functional as F
import time
from utils import *
#from loss import ClassificationLoss

# parameters of the dataset
data_folder = './data/'  # the path to the dataset
dataname = 'aifenge'  # the name of the dataset

# learning parameters
checkpoint = './results/tnet.pth'  # the path to the pretrained model，if there is no pretrained model, then modify the checkpoint as None
batch_size = 128  # batch size
start_epoch = 146  # start epoch
epochs = 300  # the number of epochs
workers = 4  # the number of workers
lr = 0.00001  # learning rate
weight_decay = 0.00001  # weight decay

# parameters of the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 4  # the number of gpus
cudnn.benchmark = True  # accelerate the cnn

# log
writer = SummaryWriter()


def main():
    """
    train.
    """
    global checkpoint, start_epoch, writer

    # initiate the model
    model = tnet()
    # initiate the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, betas=(0.9, 0.999),
                                 weight_decay=weight_decay)

    # transport the model to your device to train
    model = model.to(device)
    criterion = nn.MSELoss()
    criterion.to(device)

    # load the pretrained model
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['tnet'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    # load the dataset
    transforms = [
        RandomPatch(320),
        Normalize(),
        NumpyToTensor()
    ]
    train_dataset = HumanDataset(dataname, transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    # start training
    preloss = 10000000

    for epoch in range(start_epoch, epochs + 1):

        # if epoch == 23:  # adjust learning rate
        #     adjust_learning_rate(optimizer, 0.1)

        model.train()  # training mode：we can normalize samples of the batch

        loss_epoch = AverageMeter()  # loss function

        n_iter = len(train_loader)

        # processing
        for i, (imgs, trimaps_gt, alphas) in enumerate(train_loader):

            # transport the model to your device to train
            imgs = imgs.to(device)
            trimaps_gt = trimaps_gt.to(device)

            # forward propagation
            trimaps_pre = model(imgs)

            # calculate the loss
            loss = criterion(trimaps_pre, trimaps_gt)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()

            # renew the model
            optimizer.step()

            # log the loss
            loss_epoch.update(loss.item(), imgs.size(0))

            # monitor changes of images
            if i == n_iter - 2:
                trimaps_pre_temp = trimaps_pre[:4, :3, :, :]
                writer.add_image('TNet/epoch_' + str(epoch) + '_1',
                                 make_grid(imgs[:4, :, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image('TNet/epoch_' + str(epoch) + '_2', make_grid(trimaps_pre_temp, nrow=4, normalize=True),
                                 epoch)
                writer.add_image('TNet/epoch_' + str(epoch) + '_3',
                                 make_grid(trimaps_gt[:4, :, :, :].float().cpu(), nrow=4, normalize=True), epoch)

            # print the result
            print("We have finished " + str(i + 1) + " batches.")

        # release the memory
        del imgs, trimaps_pre, trimaps_gt, alphas, trimaps_pre_temp
        print('We have finished ' + str(epoch) + ' epochs.')

        # monitor changes of the loss
        writer.add_scalar('pretrain_tnet/Loss', loss_epoch.val, epoch)

        # save the pretrained model
        if loss_epoch.val < preloss:
            preloss = loss_epoch.val
            print("save the pretrained model\n")
            torch.save({
                'epoch': epoch,
                'tnet': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }, 'results/tnet.pth')

    # end up the training and close the writer
    writer.close()


if __name__ == '__main__':
    main()