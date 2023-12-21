import argparse
from proto.train import Train


def setup_argparse():
    parser = argparse.ArgumentParser(description='Train Prototypical network for GAN labeling')

    parser.add_argument('--train_path', type=str, default='../data/KLGradesGANs/all_train.csv',
                        help='Path to training set images')
    parser.add_argument('--valid_path', type=str, default='../data/KLGradesGANs/all_valid.csv',
                        help='Path to validation set images')
    parser.add_argument('--cuda', type=bool, default=True, action=argparse.BooleanOptionalAction, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--k', type=int, default=5, help='Value of K for K-shot learning')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--num_tasks', type=int, default=500, help='number of tasks for few-shot learning')
    parser.add_argument('--epochs', type=int, default=200, help='number of iterations')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=224, help='image resolution')
    parser.add_argument('--ckpt', type=str, help='checkpoint weight path if have one')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = setup_argparse()
    trainer = Train(args)
    trainer.train_model()

