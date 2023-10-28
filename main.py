import torch
import torch.nn as nn
import os, copy, time, shutil, socket, logging, argparse, warnings
import numpy as np
import pprint as pp
import utils.utils as utils
from os.path import join

pid = os.getpid()
hostName = socket.gethostname()
warnings.filterwarnings('ignore')

import time
import torchvision
from model import get_model
from config.defaults import _C as cfg
from pytorch_lightning import seed_everything
from active import get_strategy
from dataset import ASDADataset, build_transforms
from utils.loss_functions import ChainLoss
from utils.utils import mkidr_folder


def run_active_adaptation(cfg, source_model, trained_generator, src_dset, num_classes, device, args):
    transforms = build_transforms(cfg, 'target')
    tgt_dset = ASDADataset(cfg, cfg.DATASET.NAME, cfg.DATASET.TARGET_DOMAIN, cfg.DATASET.ROOT,
                           cfg.DATASET.NUM_CLASS, cfg.DATALOADER.BATCH_SIZE, cfg.DATALOADER.NUM_WORKERS, transforms)
    target_test_loader = tgt_dset.get_loaders()[2]

    sampling_strategy = get_strategy(cfg.ADA.AL, src_dset, tgt_dset, source_model, trained_generator, device, num_classes, cfg)
    sampling_strategy.init_idx(tgt_dset)
    start_epoch = sampling_strategy.resume_checkpoint(cfg.TRAINER.RESUME, tgt_dset)
    del source_model
    if cfg.TRAINER.EVAL_ACC:
        transfer_perf = sampling_strategy.test(target_test_loader)
        logging.info('source only performance (Before {}): Task={:.2f}'.format(cfg.ADA.DA, transfer_perf))
    else:transfer_perf = 0.
    start_perf, best_perf, best_epcoh = 0., transfer_perf if cfg.TRAINER.RESUME is None else 0., 0
    target_dir = mkidr_folder(args, cfg) if cfg.TRAINER.RESUME is None else join(*cfg.TRAINER.RESUME.split('/')[:-1])
    best_ckpt_tmp, last_ckpt_tmp = join(target_dir,'best_temp.pth'), join(target_dir,'last_temp.pth')
    if cfg.TRAINER.RESUME is not None:
        best_ckpt_tmp, last_ckpt_tmp = join(target_dir,'best_resume_temp.pth'), join(target_dir,'last_resume_temp.pth')

    # Main Active DA loop
    logging.info('------------------------------------------------------')
    model_init = 'source' if cfg.TRAINER.TRAIN_ON_SOURCE else 'scratch'
    logging.info('Running strategy: Init={} AL={} DA={}'.format(model_init, cfg.ADA.AL, cfg.ADA.DA))
    logging.info('------------------------------------------------------')

    # Instantiate active sampling strategy
    for epoch in range(start_epoch, cfg.TRAINER.MAX_EPOCHS):
        resume = 'Resuming |' if cfg.TRAINER.RESUME is not None else None
        curr_budget, used_budget = sampling_strategy.acquire_budget(epoch)
        if curr_budget > 0:
            logging.info('{} Epoch {}: selecting instances...'.format(resume, epoch))
            idxs, data, target_query = sampling_strategy.query(curr_budget, epoch)
            sampling_strategy.update(idxs, data, target_query, epoch)
        else:
            logging.info('{} Epoch {}: no budget for current epoch, skipped...'.format(resume, epoch))
        target_model = sampling_strategy.train(epoch=epoch)
        logging.info('{} Epoch {}: save last checkpoint'.format(resume, epoch))
        sampling_strategy.save_checkpoint(epoch, last_ckpt_tmp)

        logging.info('{} Epoch {}: start test performance of this epoch model'.format(resume, epoch))
        test_perf = sampling_strategy.test(target_test_loader)
        out_str = '{} Epoch {}: test performance {:.2f}'.format(resume, epoch, test_perf)
        logging.info(out_str)
        if test_perf > best_perf:
            logging.info('{} Epoch {}: acquire higher performance model {:.3f}->{:.3f} and save'.format(resume, epoch, best_perf,test_perf))
            best_perf, best_epcoh, best_target_model = test_perf, epoch, copy.deepcopy(target_model)
            sampling_strategy.save_checkpoint(best_epcoh, best_ckpt_tmp)
        logging.info('------------------------------------------------------')

    best_ckpt = join(target_dir,'best-epcoh={}_best-perf={}.pth'.format(best_epcoh, best_perf))
    last_ckpt = join(target_dir,'last-epcoh.pth')
    if cfg.TRAINER.RESUME is not None:
        best_ckpt = join(target_dir, 'best-resume-{}-epcoh={}_best-perf={}.pth'.format(args.experiment, best_epcoh, best_perf))
        last_ckpt = join(target_dir, 'last-resume-{}-epcoh.pth'.format(args.experiment))
    shutil.move(best_ckpt_tmp, best_ckpt)
    shutil.move(last_ckpt_tmp, last_ckpt)


def ADAtrain(cfg, args, task, device):
    if cfg.TRAINER.RESUME is not None: logging.info("Resuming task: {}".format(cfg.TRAINER.RESUME))
    else:logging.info("Running task: {}".format(task))

    transforms = build_transforms(cfg, 'source')
    src_dset = ASDADataset(
        cfg, cfg.DATASET.NAME, cfg.DATASET.SOURCE_DOMAIN, data_dir=cfg.DATASET.ROOT, num_classes=cfg.DATASET.NUM_CLASS,
        batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, transforms=transforms
    )

    src_train_loader, src_valid_loader, src_test_loader = src_dset.get_loaders(
        valid_type=cfg.DATASET.SOURCE_VALID_TYPE,
        valid_ratio=cfg.DATASET.SOURCE_VALID_RATIO
    )

    # model
    source_model = get_model(
        cfg.MODEL.BACKBONE.NAME, num_cls=cfg.DATASET.NUM_CLASS, normalize=cfg.MODEL.NORMALIZE,
        temp=cfg.MODEL.TEMP, botten_neck=cfg.MODEL.BOTTEN_NECK).to(device)

    if cfg.TRAINER.TRAIN_ON_SOURCE and cfg.TRAINER.MAX_SOURCE_EPOCHS > 0:
        source_path, best_source_path = utils.GetInfo_Source(cfg)
        if cfg.TRAINER.RESUME is None and cfg.TRAINER.LOAD_FROM_CHECKPOINT and os.path.exists(source_path):
            logging.info('Loading source checkpoint: {}'.format(source_path))
            source_model.load_state_dict(torch.load(source_path, map_location=device), strict=False)
            best_source_model = source_model
        elif cfg.TRAINER.RESUME is not None:
            logging.info('Loading resume checkpoint: {}'.format(cfg.TRAINER.RESUME))
            checkpoint = torch.load(cfg.TRAINER.RESUME)
            source_model.load_state_dict(checkpoint['net'], strict=False)
            best_source_model = source_model
        else:
            logging.info('Training {} model...'.format(cfg.DATASET.SOURCE_DOMAIN))
            best_val_acc, best_source_model = 0.0, None
            source_optimizer = utils.get_optim(
                cfg.OPTIM.SOURCE_NAME,
                source_model.parameters_network(cfg.OPTIM.SOURCE_LR, cfg.OPTIM.BASE_LR_MULT),
                lr=cfg.OPTIM.SOURCE_LR)
            loss_func = ChainLoss()

            for epoch in range(cfg.TRAINER.MAX_SOURCE_EPOCHS):
                utils.train(source_model, device, train_loader=src_train_loader, optimizer=source_optimizer, epoch=epoch, loss_function=loss_func, cfg=cfg)
                if epoch > 70:
                    val_acc, _ = utils.test(source_model,device,test_loader=src_valid_loader, split="source valid")
                    logging.info('[Epoch: {}] Valid Accuracy: {:.3f} '.format(epoch, val_acc))
                    if (val_acc > best_val_acc):
                        best_val_acc, best_source_model = val_acc, copy.deepcopy(source_model)
                        torch.save(best_source_model.state_dict(), best_source_path)
            del source_model
            shutil.move(best_source_path, source_path)
    else: best_source_model = source_model

    # Evaluate on source test set
    if cfg.TRAINER.EVAL_ACC:
        test_acc, _ = utils.test(best_source_model, device, src_test_loader, split="source test")
        logging.info('{} Test Accuracy: {:.3f} '.format(cfg.DATASET.SOURCE_DOMAIN, test_acc))

    if cfg.TRAINER.SOURCEFREE:
        generator = get_model(cfg.MODEL.GENERATOR, input_dim=cfg.TRAINER.SF_GENERATOR_DIM, input_size=224,
            num_cls=cfg.DATASET.NUM_CLASS, botten_neck=cfg.MODEL.BOTTEN_NECK).to(device)
        generator_path, trained_generator_path = utils.GetInfo_SourceFree(cfg)

        if cfg.TRAINER.LOAD_FROM_CHECKPOINT and os.path.exists(generator_path):
            logging.info('Loading generator checkpoint: {}'.format(generator_path))
            generator.load_state_dict(torch.load(generator_path, map_location=device), strict=False)
            trained_generator = copy.deepcopy(generator)
        else:
            logging.info('Training {} generator...'.format(cfg.DATASET.SOURCE_DOMAIN))
            generator_optimizer = utils.get_optim(cfg.OPTIM.SOURCE_NAME, generator.parameters(), lr=cfg.OPTIM.GENERATOR_LR)
            for epoch in range(cfg.TRAINER.MAX_SOURCEFREE_EPOCHS):
                utils.train_generator(generator,best_source_model,generator_optimizer,device,epoch,ChainLoss(),cfg)
            trained_generator = copy.deepcopy(generator)
            torch.save(trained_generator.state_dict(), trained_generator_path)
            shutil.move(trained_generator_path, generator_path)
        del generator
    else:
        trained_generator = None

    logging.info('Begin with Activate learing for eye domain adaptation')
    run_active_adaptation(cfg,best_source_model,trained_generator, src_dset, cfg.DATASET.NUM_CLASS, device,args)


def main():
    # repeatability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Optimal Budget Allocation for Active Domain Adaptation')
    parser.add_argument('--cfg', default='configs/visda.yaml', metavar='FILE', help='path to config file', type=str)
    parser.add_argument('--timestamp', default=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), type=str)
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--log', default='./log', type=str, help='logging directory')
    parser.add_argument('--experiment', default='add_pseudo_memorybank_no-classifier-optim', type=str)
    parser.add_argument('--nolog', action='store_true', help='whether use logger')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if not os.path.exists('./checkpoints'): os.mkdir('./checkpoints')
    target_dir = os.path.join('./checkpoints', 'target')
    if not os.path.exists(target_dir): os.mkdir(target_dir)
    if cfg.TRAINER.RESUME is None:
        cld_rt = "{}_{}_{}_{}".format(args.timestamp, cfg.DATASET.NAME, cfg.ADA.AL, cfg.ADA.DA)
        logger,args.log="{}_{}_{}_{}".format(args.timestamp,cfg.DATASET.NAME,cfg.ADA.AL,cfg.ADA.DA),join(target_dir,cld_rt)
        utils.init_logger(logger, dir=args.log)
        logging.info("Running a new experiment on {} gpu={} pid={}".format(hostName, args.gpu, pid))
        logging.info(cfg)
    else:
        foldername, _ = cfg.TRAINER.RESUME.split('/')
        args.log, logger = join(target_dir, foldername), foldername
        utils.init_logger(logger, dir=args.log)
        logging.info('------------------------------------------------------\n\n\n')
        logging.info("Resuming a experiment: <<<< {} >>>> gpu={} pid={}".format(args.experiment, args.gpu, pid))
        cfg.TRAINER.RESUME = join(target_dir, cfg.TRAINER.RESUME)
    logging.info('------------------------------------------------------')

    if type(cfg.SEED) is tuple or type(cfg.SEED) is list:seeds = cfg.SEED
    else:seeds = [cfg.SEED]
    device = torch.device("cuda:" + args.gpu) if torch.cuda.is_available() else torch.device("cpu")

    for seed in seeds:
        seed_everything(seed)
        if cfg.ADA.TASKS is not None: ada_tasks = cfg.ADA.TASKS
        else: ada_tasks = [[source, target] for source in cfg.DATASET.SOURCE_DOMAINS for target in cfg.DATASET.TARGET_DOMAINS if source != target]

        for [source, target] in ada_tasks:
            cfg.DATASET.SOURCE_DOMAIN, cfg.DATASET.TARGET_DOMAIN = source, target

            cfg.freeze()
            ADAtrain(cfg, args, task=source + '-->' + target, device=device)
            cfg.defrost()


if __name__ == '__main__':
    main()