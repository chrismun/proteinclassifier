import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import provider
import importlib
import shutil
import hydra
import omegaconf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR


# Hyperparameters
rand_rotate = False
rand_jitter = True
jitter_amount = 0.01

# My Dataset 
class ProteinDataset(Dataset):

    def __init__(self, fn):
        # data loading
        self.data = np.load(fn, 'rb')
        self.x = self.data['x']
        self.y = np.reshape(self.data['y'], (-1,))
        self.n_points = self.x.shape[1]
        self.num_classes = np.unique(self.y).shape[0]
        self.nsamples = self.x.shape[0]

    def __getitem__(self, index):
        point_data = self.x[index]

        # random rotation
        if rand_rotate:
            theta1 = np.random.uniform(0,np.pi*2)
            theta2 = np.random.uniform(0,np.pi*2)
            theta3 = np.random.uniform(0,np.pi*2)
            trans_matrix1 = np.array([[np.cos(theta1),-np.sin(theta1)],
                                        [np.sin(theta1), np.cos(theta1)]])
            trans_matrix2 = np.array([[np.cos(theta2), -np.sin(theta2)],
                                        [np.sin(theta2), np.cos(theta2)]])
            trans_matrix3 = np.array([[np.cos(theta3),-np.sin(theta3)],
                                        [np.sin(theta3), np.cos(theta3)]])
            point_data[:,[0,1]] = point_data[:,[0,1]].dot(trans_matrix1)
            point_data[:,[0,2]] = point_data[:,[0,2]].dot(trans_matrix2)
            point_data[:,[1,2]] = point_data[:,[1,2]].dot(trans_matrix3)
        # jitter
        if rand_jitter:
            jitter = np.random.normal(0,jitter_amount,size=point_data.shape)
            point_data += jitter
        return point_data, self.y[index]


    def __len__(self):
        return self.nsamples


def test(model, loader, num_class=9):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        # target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points.float())
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # print(args.pretty())

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    dataset = ProteinDataset('/ocean/projects/bio230029p/chrismun/data/surface1024.npz')
    # dataset = ProteinDataset('/ocean/projects/bio230029p/chrismun/data/surface2048.npz')
    # dataset = ProteinDataset('/ocean/projects/bio230029p/chrismun/data/trimmed_surface1024x4.npz')
    # dataset = ProteinDataset('/ocean/projects/bio230029p/chrismun/data/downsampled-5000x4.npz')


    train_dataset, test_dataset = train_test_split(dataset, test_size=0.20, random_state=12)

    # Weighted sampler
    train_class_counts = {}
    for _, label in train_dataset:
        if label in train_class_counts:
            train_class_counts[label] += 1
        else:
            train_class_counts[label] = 1

    train_samples_weight = np.array([1 / train_class_counts[y] for _, y in train_dataset])
    train_samples_weight = torch.tensor(train_samples_weight)
    train_sampler = WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), len(train_samples_weight))
    batch_size = 64

    # Dataloaders 
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler, drop_last=True)
    testDataLoader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=1, drop_last=True)

    '''MODEL LOADING'''
    args.num_class = 9
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # classifier = nn.DataParallel(classifier, [0])

    # try:
    #     checkpoint = torch.load('best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     logger.info('Use pretrain model')
    # except:
    #     logger.info('No existing model, starting training from scratch...')
    #     start_epoch = 0
    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=0.01) # sgd
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    epoch_test_acc = []
    epoch_train_acc = []
    # epoch_test_loss = []
    epoch_class_acc = []
    epoch_train_loss = []
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        classifier.train()
        test_loss = 0
        count = 0
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            # target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            # print("pred =", pred, "target =", target)
            # print("pred shape =", pred.shape, "target shape =", target.shape)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            count += batch_size
            test_loss += loss.item() * batch_size

            optimizer.step()
            global_step += 1
            
        scheduler.step()

        epoch_train_loss.append(1.0*test_loss / count)

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        epoch_train_acc.append(train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)
            epoch_test_acc.append(instance_acc)
            epoch_class_acc.append(class_acc)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    print("epoch train loss: ", epoch_train_loss)
    print("epoch test acc: ", epoch_test_acc)
    print("epoch train acc: ", epoch_train_acc)
    print("epoch class acc: ", epoch_class_acc)

if __name__ == '__main__':
    main()