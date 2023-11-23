import argparse
from siamese.train import Train


def setup_argparse():
    parser = argparse.ArgumentParser(description='Train Siamese network for GAN labeling')

    parser.add_argument('--train_path', type=str, default='data/KLGradesGANs/train',
                        help='Path to training set images')
    parser.add_argument('--valid_path', type=str, default='data/KLGradesGANs/valid',
                        help='Path to validation set images')
#     parser.add_argument('--cuda', type=bool, default=True, action=argparse.BooleanOptionalAction, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--epochs', type=int, default=50000, help='number of iterations')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, help='checkpoint weight path if have one')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = setup_argparse()
    trainer = Train(args)
    trainer.train_model()

