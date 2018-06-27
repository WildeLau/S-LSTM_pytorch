from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()
    return args
