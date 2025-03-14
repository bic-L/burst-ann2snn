import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from vedaseg.runners import TestRunner
from vedaseg.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Test a segmentation model')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('--distribute', default=False, action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--snn', type=bool, default=False)
    parser.add_argument('--timestep', type=int, default=64)
    parser.add_argument('--maxspike', type=int, default=4)
    parser.add_argument('--calib', type=str, default='light')
    parser.add_argument('--search', type=bool, default=False)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    print(args)

    _, fullname = os.path.split(cfg_path)
    fname, ext = os.path.splitext(fullname)

    root_workdir = cfg.pop('root_workdir')
    workdir = os.path.join(root_workdir, fname)
    os.makedirs(workdir, exist_ok=True)
    train_cfg = cfg['train']
    test_cfg = cfg['test']
    inference_cfg = cfg['inference']
    common_cfg = cfg['common']
    common_cfg['workdir'] = workdir
    common_cfg['distribute'] = args.distribute
    common_cfg['snn'] = args.snn
    common_cfg['timestep'] = args.timestep
    common_cfg['maxspike'] = args.maxspike

    runner = TestRunner(train_cfg, test_cfg, inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    runner()


if __name__ == '__main__':
    main()
