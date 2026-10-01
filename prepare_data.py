"""Fetch the official torchvision MNIST archives into a reusable data directory."""
import argparse


def main(args):
    from torchvision.datasets import MNIST
    for training in (True, False):
        dataset = MNIST(args.root, train=training, download=True)
        print(f'{"train" if training else "test"}: {len(dataset)} examples under {dataset.raw_folder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='datasets')
    main(parser.parse_args())
