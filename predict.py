"""Classify image files with a trained OSLGN checkpoint on CUDA."""
import argparse

import numpy as np
import torch
from PIL import Image

from utils.experiment import restore_model
from utils.util import write_json


def main(args):
    model, config = restore_model(args.checkpoint, args.device)
    if config['model']['input_dim'] != 784:
        raise ValueError('Image-file prediction requires an MNIST-size 784-input checkpoint')
    arrays = []
    for path in args.images:
        with Image.open(path) as image:
            array = np.asarray(image.convert('L').resize((28, 28)), dtype=np.float32) / 255.0
        arrays.append(array.reshape(-1))
    images = torch.as_tensor(np.stack(arrays), device=args.device)
    images = (images > config['data']['threshold']).float()
    with torch.no_grad():
        features = model.logic_features(images)
        logits = model.head(features)
        predictions = logits.argmax(-1).tolist()
    write_json(args.output, {'images': args.images, 'predictions': predictions,
                            'logic_features': features.tolist(), 'logits': logits.tolist()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/predictions.json')
    main(parser.parse_args())
