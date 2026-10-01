"""CUDA event timings for hard logic layers and the classifier head."""

import argparse
import math

import torch

from experiments.wiring import cuda_device, restore_model, write_json


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('At least one timing observation is required')
    def percentile(fraction):
        location = fraction * (len(ordered) - 1)
        low = math.floor(location)
        high = math.ceil(location)
        return ordered[low] + (ordered[high] - ordered[low]) * (location - low)
    mean = sum(ordered) / len(ordered)
    return dict(
        repeats=len(ordered),
        mean_ms=mean,
        minimum_ms=ordered[0],
        median_ms=percentile(.5),
        p90_ms=percentile(.9),
        p99_ms=percentile(.99),
        maximum_ms=ordered[-1],
        std_ms=math.sqrt(sum((value - mean) ** 2 for value in ordered) / len(ordered)),
    )


@torch.no_grad()
def timed(operation, warmup, repeats, device):
    if warmup < 0 or repeats < 1:
        raise ValueError('Warmup is nonnegative and repetitions must be positive')
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measurements = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        stop.record()
        stop.synchronize()
        measurements.append(start.elapsed_time(stop))
        del result
    return dict(
        **distribution(measurements),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


@torch.no_grad()
def benchmark(model, batch_sizes, warmup, repeats, device, seed):
    if not batch_sizes or min(batch_sizes) < 1 or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError('Batch sizes must be distinct positive integers')
    generator = torch.Generator(device=device).manual_seed(seed)
    records = []
    with torch.cuda.device(device):
        for batch in batch_sizes:
            values = torch.randint(0, 2, (batch, model.input_dim), device=device, generator=generator).float()
            full = timed(lambda: model(values), warmup, repeats, device)
            layers = []
            previous = values
            for index, layer in enumerate(model.layers):
                inputs = previous
                report = timed(lambda layer=layer, inputs=inputs: layer(inputs), warmup, repeats, device)
                previous = layer(inputs)
                layers.append(dict(layer=index, input_dim=inputs.shape[1], output_dim=previous.shape[1], **report))
            features = previous
            head = timed(lambda: model.head(features), warmup, repeats, device)
            records.append(dict(
                batch_size=batch,
                full=full,
                samples_per_second=1000 * batch / full['mean_ms'],
                layers=layers,
                head=head,
            ))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 64, 256])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--packed', action='store_true')
    args = parser.parse_args()
    device = cuda_device(args.device)
    model, config = restore_model(args.checkpoint, device)
    if args.packed:
        results = benchmark_packed(model, args.batch_sizes, args.warmup, args.repeats, device, args.seed)
    else:
        results = benchmark(model, args.batch_sizes, args.warmup, args.repeats, device, args.seed)
    write_json(args.output, dict(
        checkpoint=args.checkpoint,
        device=torch.cuda.get_device_name(device),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        model=config['model'],
        input_distribution='independent Bernoulli(0.5) bits',
        timing='CUDA events around device-resident forward passes',
        measurements=results,
    ))


class PackedCircuit:
    def __init__(self, circuit, device):
        from networks.analysis import validate_circuit
        validate_circuit(circuit)
        self.device = cuda_device(device)
        self.input_dim = circuit['input_dim']
        self.layers = []
        for layer in circuit['layers']:
            self.layers.append({name: torch.tensor(layer[name], dtype=torch.long, device=self.device)
                                for name in ('left', 'right', 'gate')})
        self.weight = torch.tensor(circuit['head']['weight'], device=self.device, dtype=torch.float32)
        self.bias = torch.tensor(circuit['head']['bias'], device=self.device, dtype=torch.float32)
        self.shifts = torch.arange(8, device=self.device, dtype=torch.long)

    def pack(self, values):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError('Packed circuit input must have N,input_dim dimensions')
        if values.device != self.weight.device:
            raise ValueError('Packed inputs and circuit must use the same CUDA device')
        if not ((values == 0) | (values == 1)).all():
            raise ValueError('Packed circuit input must be binary')
        count = len(values)
        padding = (-count) % 8
        bits = values.to(torch.long)
        if padding:
            bits = torch.cat((bits, torch.zeros(padding, self.input_dim, device=bits.device, dtype=bits.dtype)))
        reshaped = bits.reshape(-1, 8, self.input_dim)
        packed = (reshaped << self.shifts[None, :, None]).sum(1).to(torch.uint8)
        return packed, count

    def unpack(self, packed, count):
        values = ((packed.long()[:, None] >> self.shifts[None, :, None]) & 1).flatten(0, 1)
        return values[:count].float()

    def logic(self, packed):
        values = packed
        for layer in self.layers:
            left = values[:, layer['left']]
            right = values[:, layer['right']]
            inverse_left = torch.bitwise_not(left)
            inverse_right = torch.bitwise_not(right)
            gate = layer['gate'][None]
            zero = torch.zeros_like(left)
            output = torch.where((gate & 8) != 0, inverse_left & inverse_right, zero)
            output |= torch.where((gate & 4) != 0, inverse_left & right, zero)
            output |= torch.where((gate & 2) != 0, left & inverse_right, zero)
            output |= torch.where((gate & 1) != 0, left & right, zero)
            values = output
        return values

    @torch.no_grad()
    def __call__(self, values):
        packed, count = self.pack(values)
        features = self.unpack(self.logic(packed), count)
        return features @ self.weight.T + self.bias

    @torch.no_grad()
    def verify(self, model, values):
        packed, count = self.pack(values)
        features = self.unpack(self.logic(packed), count)
        expected_features = model.logic_features(values)
        if not torch.equal(features, expected_features):
            raise ValueError('Bit-packed gate features differ from the trained hard circuit')
        output = features @ self.weight.T + self.bias
        expected = model(values)
        difference = (output - expected).abs()
        return dict(
            samples=count,
            packed_rows=len(packed),
            binary_features_exact=True,
            maximum_logit_error=float(difference.max()),
            predictions_equal=bool(torch.equal(output.argmax(-1), expected.argmax(-1))),
        )


@torch.no_grad()
def benchmark_packed(model, sizes, warmup, repeats, device, seed):
    packed = PackedCircuit(model.export_circuit(), device)
    generator = torch.Generator(device=device).manual_seed(seed)
    results = []
    with torch.cuda.device(device):
        for size in sizes:
            values = torch.randint(0, 2, (size, model.input_dim), device=device, generator=generator).float()
            verification = packed.verify(model, values)
            encoded, count = packed.pack(values)
            full = timed(lambda: packed(values), warmup, repeats, device)
            gates = timed(lambda: packed.logic(encoded), warmup, repeats, device)
            original = timed(lambda: model(values), warmup, repeats, device)
            results.append(dict(
                batch_size=count,
                packed_input_bytes=encoded.numel() * encoded.element_size(),
                float_input_bytes=values.numel() * values.element_size(),
                verification=verification,
                packed_full=full,
                packed_logic=gates,
                original_full=original,
                speed_ratio=original['mean_ms'] / full['mean_ms'],
                packed_samples_per_second=1000 * size / full['mean_ms'],
            ))
    return results
