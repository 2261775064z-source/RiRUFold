import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import get_model
from models.checkpoint import load_checkpoint
from utils.data import NUDTDataset, IRSTD1kDataset, SirstAugDataset, SirstDataset
from utils.metrics import SegmentationMetricTPFNFP
from utils.my_pd_fa import my_PD_FA


DATASET_SPECS = {
    'nudt': (['NUDT-SIRST'], NUDTDataset),
    'sirstaug': (['sirst_aug', 'SIRST-Aug'], SirstAugDataset),
    'irstd1k': (['IRSTD-1K', 'IRSTD-1k'], IRSTD1kDataset),
    'sirst': (['NUAA-SIRST', 'sirst'], SirstDataset),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained RiRUFold checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=sorted(DATASET_SPECS),
                        help='dataset to evaluate: nudt | sirstaug | irstd1k | sirst')
    parser.add_argument('--data-root', type=Path, default=Path('./datasets'),
                        help='parent directory containing benchmark datasets')
    parser.add_argument('--split-dir', type=Path, default=None,
                        help='optional custom split directory for NUDT-SIRST')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='path to trained model checkpoint (.pkl / .pth)')
    parser.add_argument('--net-name', type=str, default='rirufold', help='model name')
    parser.add_argument('--stage-num', type=int, default=5, help='number of unfolding stages K')
    parser.add_argument('--base-size', type=int, default=256, help='evaluation image size')
    parser.add_argument('--batch-size', type=int, default=4, help='evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='data loader workers')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold in (0, 1)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json', type=Path, default=None, help='optional path to save metrics JSON')
    return parser.parse_args()


def resolve_dataset_root(data_root: Path, candidates):
    for name in candidates:
        folder = data_root / name
        if folder.is_dir():
            return folder
    raise FileNotFoundError(f'Dataset folder not found under {data_root} (expected one of {candidates})')


def main():
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError('--threshold must be strictly between 0 and 1')

    device = torch.device(args.device)
    candidates, dataset_cls = DATASET_SPECS[args.dataset]
    dataset_root = resolve_dataset_root(args.data_root, candidates)

    ds_kwargs = dict(base_dir=str(dataset_root), mode='test', base_size=args.base_size)
    if args.dataset == 'nudt' and args.split_dir is not None:
        ds_kwargs['split_dir'] = args.split_dir
    dataset = dataset_cls(**ds_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.net_name, stage_num=args.stage_num).to(device)
    load_checkpoint(model, args.checkpoint, device='cpu')
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    metric = SegmentationMetricTPFNFP(nclass=1)
    detection_metric = my_PD_FA()
    logit_shift = math.log(args.threshold / (1.0 - args.threshold))

    with torch.no_grad():
        for images, labels in loader:
            _, logits = model(images.to(device))
            logits_cpu = logits.cpu()
            labels_cpu = labels.cpu()

            metric.update(labels_cpu, logits_cpu - logit_shift)
            probs = torch.sigmoid(logits_cpu).numpy()
            targets = labels_cpu.numpy()

            for prob, target in zip(probs, targets):
                prob = np.squeeze(prob)
                target = np.squeeze(target)
                binary = (prob >= args.threshold).astype(np.float32)
                if (target > 0.5).any():
                    detection_metric.update(binary, target)
                else:
                    detection_metric.background_area += target.size
                    detection_metric.false_detect += int((binary > 0.5).sum())

    iou, precision, recall, f1 = metric.get()
    pd, fa = detection_metric.get()

    result = {
        'dataset': args.dataset,
        'images': len(dataset),
        'stage_num': args.stage_num,
        'params_M': round(num_params / 1e6, 3),
        'mIoU_percent': round(float(iou) * 100.0, 2),
        'F1_percent': round(float(f1) * 100.0, 2),
        'Pd_percent': round(float(pd) * 100.0, 2),
        'Fa_1e-6': round(float(fa) * 1e6, 2),
        'Precision_percent': round(float(precision) * 100.0, 2),
        'Recall_percent': round(float(recall) * 100.0, 2),
    }

    print(
        f"[{args.dataset.upper()}] Images: {len(dataset)} | Params: {num_params / 1e6:.3f}M | "
        f"mIoU: {iou * 100:.2f}% | F1: {f1 * 100:.2f}% | "
        f"Pd: {pd * 100:.2f}% | Fa: {fa * 1e6:.2f} x 10^-6"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f'Saved evaluation summary to {args.output_json}')


if __name__ == '__main__':
    main()
