import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import logging
from tqdm import tqdm

import scipy.io as scio

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
logger_init = False


def train(model, device, train_loader, optimizer, epoch, loss_function, cfg):
    model.train()
    total_loss, correct = 0.0, 0
    for batch_idx, (data, target, _) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, emb = model(data, True)
        loss = nn.CrossEntropyLoss()(output, target)

        if epoch>5:
            try:
                loss += loss_function(target, emb, device)
            except:
                logging.info('error XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        corr = pred.eq(target.view_as(pred)).sum().item()
        correct += corr
        loss.backward()
        optimizer.step()

    train_acc = 100. * correct / len(train_loader.sampler)
    avg_loss = total_loss / len(train_loader.sampler)
    logging.info('Train Epoch: {} | Avg. Loss: {:.3f} | Train Acc: {:.3f}'.format(epoch, avg_loss, train_acc))
    return avg_loss


def test(model, device, test_loader, mat=False, epoch=None, split="target test"):
    model.eval()
    test_loss = 0
    correct = 0
    start_test = True
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            softmax_score, softmax_score_id = nn.Softmax(dim=1)(output).max(dim=1)
            energy_score = -1 * torch.logsumexp(-1 * output, dim=1, keepdim=False)

            if mat:
                if start_test:
                    all_softmax_score_id = softmax_score_id.float().cpu()
                    all_softmax_score = softmax_score.float().cpu()
                    all_label = target.float().cpu()
                    all_energy_score = energy_score.float().cpu()
                    start_test = False
                else:
                    all_softmax_score_id = torch.cat((all_softmax_score_id, softmax_score_id.float().cpu()), 0)
                    all_softmax_score = torch.cat((all_softmax_score, softmax_score.float().cpu()), 0)
                    all_label = torch.cat((all_label, target.float().cpu()), 0)
                    all_energy_score = torch.cat((all_energy_score, energy_score.float().cpu()), 0)


            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr
            del loss, output

        if mat:
            train_dict = {}
            train_dict['true_labels'] = all_label.numpy()
            train_dict['pseudo_labels'] = all_softmax_score_id.numpy()
            train_dict['energy'] = all_energy_score.numpy()
            train_dict['confidence'] = all_softmax_score.numpy()
            scio.savemat('./checkpoints/vis/' + str(epoch) + '_E.C.mat', train_dict)

    test_loss /= len(test_loader.sampler)
    test_acc = 100. * correct / len(test_loader.sampler)

    return test_acc, test_loss


def train_generator(generator, source_model, optim, device, epoch, loss_function, cfg):
    z = torch.rand(cfg.TRAINER.SF_GENERATOR_BZ, cfg.TRAINER.SF_GENERATOR_DIM).to(device)
    labels = torch.randint(0, cfg.DATASET.NUM_CLASS, (cfg.TRAINER.SF_GENERATOR_BZ,)).to(device)

    images_feas = generator(z, labels)
    output_teacher_batch = source_model.classifier(images_feas)

    loss_ce = torch.nn.CrossEntropyLoss()(output_teacher_batch, labels)
    if epoch >= 40:
        loss_chain = loss_function(labels, images_feas, device)
        if epoch % 20 == 0:
            logging.info('Train Epoch: {} | CE. Loss: {:.3f} | Chain. Loss: {:.3f}'.format(epoch, loss_ce, loss_chain))
        loss = loss_ce + loss_chain
    else:
        if epoch % 20 == 0:
            logging.info('Train Epoch: {} | CE. Loss: {:.3f}'.format(epoch, loss_ce))
        loss = loss_ce

    # loss of Generator
    optim.zero_grad()
    loss.backward()
    optim = exp_lr_scheduler(optim, step=epoch, cfg=cfg)
    optim.step()


def get_optim(name, *args, **kwargs):
    if name == 'Adadelta':
        return optim.Adadelta(*args, **kwargs)
    elif name == 'Adam':
        return optim.Adam(*args, **kwargs)
    elif name == 'SGD':
        return optim.SGD(*args, **kwargs, momentum=0.9, nesterov=True)


def exp_lr_scheduler(optim, step, cfg, lr_decay_step=2000, step_decay_weight=0.95, ):
    current_lr = cfg.OPTIM.GENERATOR_LR * (step_decay_weight ** (step / lr_decay_step))
    for param_group in optim.param_groups:
        param_group['lr'] = current_lr
    return optim


def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)  # F.normalize只能处理两维的数据，L2归一化
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())  # 计算余弦相似度
    return similarity  # 返回余弦相似度

def GetInfo_Source(cfg):
    source_file = '{}_{}_{}_source_{}_{}.pth'.format(cfg.DATASET.SOURCE_DOMAIN,
                                                  str(cfg.DATASET.CROP_SIZE),
                                                  cfg.MODEL.BACKBONE.NAME,
                                                  cfg.TRAINER.SOURCE_MODE,
                                                  cfg.TRAINER.MAX_SOURCE_EPOCHS)
    source_dir = os.path.join('checkpoints', 'source')
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    source_path = os.path.join(source_dir, source_file)
    best_source_file = '{}_{}_{}_source_best_{}_{}.pth'.format(cfg.DATASET.SOURCE_DOMAIN,
                                                            str(cfg.DATASET.CROP_SIZE),
                                                            cfg.MODEL.BACKBONE.NAME,
                                                            cfg.TRAINER.SOURCE_MODE,
                                                            cfg.TRAINER.MAX_SOURCE_EPOCHS)
    best_source_path = os.path.join(source_dir, best_source_file)
    return source_path, best_source_path


def GetInfo_SourceFree(cfg):
    generator_file = '{}_{}_{}_{}_generator_{}_{}.pth'.format(
        cfg.DATASET.SOURCE_DOMAIN,
        str(cfg.DATASET.CROP_SIZE),
        cfg.MODEL.BACKBONE.NAME,
        cfg.TRAINER.SOURCE_MODE,
        cfg.TRAINER.GENERATOR_TYPE,
        cfg.TRAINER.MAX_SOURCEFREE_EPOCHS
    )
    generator_dir = os.path.join('checkpoints', 'generator')
    if not os.path.exists(generator_dir):
        os.makedirs(generator_dir)
    generator_path = os.path.join(generator_dir, generator_file)
    best_generator_file = '{}_{}_{}_{}_generator_{}_best_{}.pth'.format(
      cfg.DATASET.SOURCE_DOMAIN,
      str(cfg.DATASET.CROP_SIZE),
      cfg.MODEL.BACKBONE.NAME,
      cfg.TRAINER.SOURCE_MODE,
      cfg.TRAINER.GENERATOR_TYPE,
      cfg.TRAINER.MAX_SOURCEFREE_EPOCHS
    )
    trained_generator_path = os.path.join(generator_dir, best_generator_file)
    return generator_path, trained_generator_path


def init_logger(_log_file, dir='log/'):
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H.%M.%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)

    if _log_file is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        log_file = os.path.join(dir, _log_file + '.log')
        fhlr = logging.FileHandler(log_file)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    global logger_init
    logger_init = True

def mkidr_folder(args, cfg):
    target_dir = os.path.join('checkpoints', 'target')
    if not os.path.exists(target_dir): os.mkdir(target_dir)
    child_root = "{}_{}_{}_{}".format(args.timestamp, cfg.DATASET.NAME, cfg.ADA.AL, cfg.ADA.DA)
    target_dir = os.path.join(target_dir, child_root)
    if not os.path.exists(target_dir): os.mkdir(target_dir)
    return target_dir

