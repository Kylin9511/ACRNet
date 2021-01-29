import os
import random
import torch

from model import acrnet
from utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device


def init_model(args, print_arch=True):
    # Model loading
    model = acrnet(reduction=args.reduction,
                   expansion=args.expansion)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained), args.pretrained
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model info logging
    logger.info(f'=> Model Name: ACRNet')
    logger.info(f'=> Model Config: compression ratio=1/{args.reduction}; expansion={args.expansion}')
    if print_arch is True:
        logger.info(f'\n{line_seg}\n{model}\n{line_seg}\n')

    return model
