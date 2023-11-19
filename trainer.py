import argparse
from train import Train
from fid import eval_fid

import argparse
from train import Train
from fid import eval_fid

def setup_argparse():
    parser = argparse.ArgumentParser(description='Knee KL Grade GAN')

    parser.add_argument('--path', type=str, default='../data/KLGradeGANs/10-shot_train.txt',
                        help='Path to text file containing training set images')
    parser.add_argument('--kl_grade', type=str, default=None, help="KL grade to train GAN for. Ex. KL0, KL1, KL2, ...")
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, help='checkpoint weight path if have one')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = setup_argparse()
    trainer = Train(args)
    trainer.train_model()

