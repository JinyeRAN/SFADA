from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.SEED = 0
_C.NOTE = ''

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ''
_C.DATASET.NAME = ''
# List of domains
_C.DATASET.SOURCE_DOMAINS = []
_C.DATASET.TARGET_DOMAINS = []
_C.DATASET.SOURCE_DOMAIN = ''
_C.DATASET.TARGET_DOMAIN = ''
_C.DATASET.CROP_SIZE = 224
_C.DATASET.RESIZE_SIZE = 256
_C.DATASET.SOURCE_VALID_TYPE = 'val'
_C.DATASET.SOURCE_VALID_RATIO = 1.0
_C.DATASET.SOURCE_TRANSFORMS = ('Resize','RandomHorizontalFlip','RandomCrop','Normalize')
_C.DATASET.TARGET_TRANSFORMS = ('Resize','RandomHorizontalFlip','RandomCrop','Normalize')
_C.DATASET.QUERY_TRANSFORMS = ('Resize','CenterCrop','Normalize')
_C.DATASET.TEST_TRANSFORMS = ('Resize','CenterCrop','Normalize')
_C.DATASET.RAND_TRANSFORMS = 'rand_transform'
_C.DATASET.NUM_CLASS = 5

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE = 32

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.INIT_WEIGHTS = ''
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ''
_C.MODEL.BACKBONE.PRETRAINED = True
_C.MODEL.NORMALIZE = False
_C.MODEL.TEMP = 0.05
_C.MODEL.BOTTEN_NECK = 256

_C.MODEL.GENERATOR = 'Generator'

_C.MODEL.LAMBDA= 0.25
###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = 'Adadelta'
_C.OPTIM.SOURCE_NAME = 'Adadelta'
_C.OPTIM.UDA_NAME = 'Adadelta'
_C.OPTIM.SOURCE_LR = 0.1
_C.OPTIM.UDA_LR = 0.1
_C.OPTIM.ADAPT_LR = 0.1
_C.OPTIM.BASE_LR_MULT = 0.1

_C.OPTIM.GENERATOR_LR = 0.01
###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.RESUME = None
_C.TRAINER.LOAD_FROM_CHECKPOINT = True
_C.TRAINER.TRAIN_ON_SOURCE = True
_C.TRAINER.MAX_SOURCE_EPOCHS = 20
_C.TRAINER.MAX_EPOCHS = 20
_C.TRAINER.EVAL_ACC = True
_C.TRAINER.ITER_PER_EPOCH = None
_C.TRAINER.SOURCEFREE = True
_C.TRAINER.SF_GENERATOR_DIM = 100
_C.TRAINER.SF_GENERATOR_BZ=64
_C.TRAINER.MAX_SOURCEFREE_EPOCHS=1000
_C.TRAINER.SOURCE_MODE='clear'
_C.TRAINER.GENERATOR_TYPE='ce+cl'
###########################
# Active DA
###########################
_C.ADA = CN()
_C.ADA.TASKS = None
_C.ADA.BUDGET = 0.05
_C.ADA.ROUND = 5
_C.ADA.ROUNDS = None
_C.ADA.DA = ''
_C.ADA.AL = ''

###########################
# LPDA
###########################
_C.LPDA = CN()
_C.LPDA.S_K = 10
_C.LPDA.S_TH = 0.8
_C.LPDA.A_TH = 0.8
_C.LPDA.A_RAND_NUM = 1
