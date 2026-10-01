import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data import load_dataset
from utils.experiment import experiment_config
from utils.models import OSLGN
from utils.util import cuda_device, evaluate, set_seed, write_json


def main(args):
    config = experiment_config(args)
    set_seed(config['train']['seed'])
    torch.set_num_threads(config['train'].get('num_threads', 1))
    device = cuda_device(args.device)
    print(f"Dataset: {config['data']['dataset']} | depth: {config['model']['depth']} | device: {device}")
    model = OSLGN(**config['model']).to(device)
    train_set = load_dataset(config, 'train', args.data, download=args.download)
    val_set = load_dataset(config, 'val', args.data, download=args.download)
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    output = Path(config['train']['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    initial = evaluate(model, train_loader, device)
    history, best = [], float('inf')
    for epoch in range(config['train']['epochs']):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images.to(device)), targets.to(device))
            loss.backward()
            if config['train'].get('clip_max_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_max_norm'])
            optimizer.step()
        scores = evaluate(model, val_loader, device)
        history.append({'epoch': epoch + 1, **scores})
        saved = {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1, 'metrics': scores}
        torch.save(saved, output / 'last.pth')
        if scores['loss'] < best:
            best = scores['loss']
            torch.save(saved, output / 'best.pth')
        print(f"Epoch {epoch + 1}: loss={scores['loss']:.4f}, accuracy={100 * scores['accuracy']:.2f}%")
    final = evaluate(model, train_loader, device)
    write_json(output / 'metrics.json', {'task': 'npz' if args.data else config['data']['dataset'],
               'initial_train': initial, 'final_train': final, 'history': history})
    if args.smoke and final['loss'] >= initial['loss']:
        raise RuntimeError('Smoke optimization did not decrease training loss')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'boolean'], default='mnist')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--config')
    parser.add_argument('--data', help='NPZ overrides the configured dataset loader')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
