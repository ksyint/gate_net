import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.analysis import wiring_statistics
from utils.data import load_dataset
from utils.experiment import experiment_config
from utils.models import OSLGN
from utils.util import cuda_device, evaluate, set_seed, write_json


def main(args):
    config = experiment_config(args)
    set_seed(config['train']['seed'])
    torch.set_num_threads(config['train'].get('num_threads', 1))
    device = cuda_device(args.device)
    download = not getattr(args, 'offline', False)
    print(f"Dataset: {config['data']['dataset']} | depth: {config['model']['depth']} | device: {device}")
    model = OSLGN(**config['model']).to(device)
    train_set = load_dataset(config, 'train', args.data, download=download)
    use_validation = bool(args.data) or config['data']['dataset'] != 'mnist' or config['data']['validation_fraction'] > 0
    val_set = load_dataset(config, 'val', args.data, download=download) if use_validation else None
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, pin_memory=True) if val_set is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'], betas=(0.9, 0.999))
    output = Path(config['train']['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    history, best, start_epoch = [], float('inf'), 0
    if getattr(args, 'resume', None):
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        if saved['config']['model'] != config['model'] or saved['config']['data'] != config['data']:
            raise ValueError('Resume requires matching architecture and data configuration')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = config['train']['learning_rate']
        start_epoch, best = saved['epoch'], saved.get('best_loss', float('inf'))
        history = saved.get('history', [])
        if 'torch_rng' in saved:
            torch.set_rng_state(saved['torch_rng'].cpu())
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        total_loss = correct = count = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            if config['train'].get('clip_max_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_max_norm'])
            optimizer.step()
            total_loss += loss.detach().item() * len(targets)
            correct += (logits.detach().argmax(-1) == targets).sum().item()
            count += len(targets)
        training = {'loss': total_loss / count, 'accuracy': correct / count, 'samples': count}
        scores = evaluate(model, val_loader, device) if val_loader is not None else training
        record = {'epoch': epoch + 1, 'train': training, 'wiring': wiring_statistics(model)}
        if val_loader is not None:
            record['validation'] = scores
        history.append(record)
        improved = scores['loss'] < best
        best = min(best, scores['loss'])
        saved = {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1, 'metrics': scores, 'best_loss': best, 'history': history,
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device)}
        torch.save(saved, output / 'last.pth')
        if improved:
            torch.save(saved, output / 'best.pth')
        print(f"Epoch {epoch + 1}: loss={scores['loss']:.4f}, accuracy={100 * scores['accuracy']:.2f}%", flush=True)
    report = {'config': config, 'task': 'npz' if args.data else config['data']['dataset'],
              'final_train': evaluate(model, train_loader, device), 'history': history,
              'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'wiring': wiring_statistics(model)}
    if config['data']['dataset'] == 'mnist' and not args.data:
        test_set = load_dataset(config, 'test', download=download)
        report['final_test'] = evaluate(model, DataLoader(test_set, batch_size=128), device)
    write_json(output / 'metrics.json', report)
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'boolean'], default='mnist')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--config')
    parser.add_argument('--data', help='NPZ overrides the configured dataset loader')
    parser.add_argument('--data-root', help='Directory containing MNIST/raw')
    parser.add_argument('--download', action='store_true', help='MNIST downloads automatically when online')
    parser.add_argument('--offline', action='store_true', help='Use existing MNIST files without downloads')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
