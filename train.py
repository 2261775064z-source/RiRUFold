import os
import os.path as osp
import time
import datetime
import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

from models import get_model
from models.checkpoint import load_checkpoint
from utils.data import NUDTDataset, IRSTD1kDataset, SirstAugDataset, SirstDataset
from utils.loss import SoftLoULoss
from utils.lr_scheduler import LR_Scheduler_Head
from utils.metrics import SegmentationMetricTPFNFP
from utils.logger import setup_logger


class JsonSummaryWriter:
    """Fallback scalar/text logger when tensorboardX is not installed."""

    def __init__(self, log_dir):
        self.path = osp.join(log_dir, 'scalars.jsonl')

    def _write(self, kind, tag, value, step=None):
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(type=kind, tag=tag, value=value, step=step)) + '\n')

    def add_scalar(self, tag, value, step):
        self._write('scalar', tag, float(value), int(step))

    def add_text(self, tag, value):
        self._write('text', tag, str(value))


def _resolve_dataset_dir(data_root, candidates):
    root = Path(data_root)
    for name in candidates:
        candidate = root / name
        if candidate.is_dir():
            return str(candidate)
    return str(root / candidates[0])


def parse_args():
    parser = ArgumentParser(description='PyTorch Training for RiRUFold')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='nudt',
                        choices=['nudt', 'sirstaug', 'irstd1k', 'sirst'],
                        help='benchmark dataset: nudt | sirstaug | irstd1k | sirst')
    parser.add_argument('--data-root', type=str, default='./datasets',
                        help='parent directory containing dataset folders')
    parser.add_argument('--split-dir', type=str, default=None,
                        help='optional custom split directory for NUDT-SIRST')
    parser.add_argument('--base-size', type=int, default=256, help='input image resolution')
    parser.add_argument('--crop-size', type=int, default=256, help='crop image resolution')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4, help='training batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm-up epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='learning rate scheduler')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='enable deterministic cuDNN')

    # Network parameters
    parser.add_argument('--net-name', type=str, default='rirufold', help='model name')
    parser.add_argument('--admm-stage-num', type=int, default=5, help='number of unfolding stages K')
    parser.add_argument('--init-checkpoint', type=str, default=None, help='optional pretrained checkpoint')

    # Logging & checkpoint saving
    parser.add_argument('--save-iter-step', type=int, default=0,
                        help='extra validation interval in iterations (0 = validate per epoch)')
    parser.add_argument('--log-per-iter', type=int, default=20, help='logging interval in iterations')
    parser.add_argument('--base-dir', type=str, default='./results', help='output root directory')
    parser.add_argument('--run-name', type=str, default='', help='optional custom run folder name')

    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)

    args.time_name = time.strftime('%Y%m%dT%H-%M-%S', time.localtime())
    args.folder_name = args.run_name or f'{args.time_name}_{args.net_name}_{args.dataset}'
    args.save_folder = osp.join(args.base_dir, args.folder_name)

    if args.seed != 0:
        set_seeds(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.logger = setup_logger('RiRUFold', args.save_folder, 0, filename='log.txt')
    with open(osp.join(args.save_folder, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump({k: v for k, v in vars(args).items() if k != 'logger'}, handle, indent=2)
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        if args.dataset == 'nudt':
            ds_dir = _resolve_dataset_dir(args.data_root, ['NUDT-SIRST'])
            trainset = NUDTDataset(base_dir=ds_dir, mode='train', base_size=args.base_size, split_dir=args.split_dir)
            valset = NUDTDataset(base_dir=ds_dir, mode='test', base_size=args.base_size, split_dir=args.split_dir)
            overlap = set(trainset.names) & set(valset.names)
            if overlap:
                raise ValueError(f'NUDT train/test overlap detected: {sorted(overlap)[:5]}')
        elif args.dataset == 'sirstaug':
            ds_dir = _resolve_dataset_dir(args.data_root, ['sirst_aug', 'SIRST-Aug'])
            trainset = SirstAugDataset(base_dir=ds_dir, mode='train', base_size=args.base_size)
            valset = SirstAugDataset(base_dir=ds_dir, mode='test', base_size=args.base_size)
        elif args.dataset == 'irstd1k':
            ds_dir = _resolve_dataset_dir(args.data_root, ['IRSTD-1K', 'IRSTD-1k'])
            trainset = IRSTD1kDataset(base_dir=ds_dir, mode='train', base_size=args.base_size)
            valset = IRSTD1kDataset(base_dir=ds_dir, mode='test', base_size=args.base_size)
        elif args.dataset == 'sirst':
            ds_dir = _resolve_dataset_dir(args.data_root, ['NUAA-SIRST', 'sirst'])
            trainset = SirstDataset(base_dir=ds_dir, mode='train', base_size=args.base_size)
            valset = SirstDataset(base_dir=ds_dir, mode='val', base_size=args.base_size)
        else:
            raise NotImplementedError(f'Unknown dataset: {args.dataset}')

        loader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': torch.cuda.is_available(),
        }
        self.train_data_loader = Data.DataLoader(trainset, shuffle=True, **loader_kwargs)
        self.val_data_loader = Data.DataLoader(valset, shuffle=False, **loader_kwargs)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = get_model(args.net_name, stage_num=args.admm_stage_num).to(self.device)
        if args.init_checkpoint:
            load_checkpoint(self.net, args.init_checkpoint, device='cpu')

        self.softiou = SoftLoULoss()
        self.mse = torch.nn.MSELoss()
        self.scheduler = LR_Scheduler_Head(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_data_loader), lr_step=10
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_iou = -1.0
        self.best_fmeasure = 0.0

        self.writer = (
            SummaryWriter(log_dir=args.save_folder)
            if SummaryWriter is not None
            else JsonSummaryWriter(args.save_folder)
        )
        self.logger = args.logger
        num_params = sum(p.numel() for p in self.net.parameters())
        self.logger.info(f'Device: {self.device} | Stages K={self.net.stage_num} | Params: {num_params / 1e6:.3f}M')

    def compute_loss(self, data, labels, out_background, out_target):
        loss_softiou = self.softiou(out_target, labels)
        loss_reconstruction = self.mse(out_background, data)
        loss_all = loss_softiou + 0.01 * loss_reconstruction
        return loss_all, loss_softiou, loss_reconstruction

    def training(self):
        start_time = time.time()
        log_fmt = (
            'Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || '
            'Loss: {:.4f} (SoftIoU: {:.4f}, Recon: {:.4f}) || Cost: {} || ETA: {}'
        )
        for epoch in range(self.args.epochs):
            for i, (data, labels) in enumerate(self.train_data_loader):
                self.net.train()
                self.scheduler(self.optimizer, i, epoch, self.best_iou)

                data = data.to(self.device)
                labels = labels.to(self.device)
                out_D, out_T = self.net(data)

                loss_all, loss_softiou, loss_rec = self.compute_loss(data, labels, out_D, out_T)
                if not torch.isfinite(loss_all):
                    raise FloatingPointError('Non-finite training loss encountered')

                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                self.iter_num += 1
                elapsed = time.time() - start_time
                cost_str = str(datetime.timedelta(seconds=int(elapsed)))
                eta_sec = (elapsed / self.iter_num) * (self.max_iter - self.iter_num)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                self.writer.add_scalar('Train/Loss_All', float(loss_all.item()), self.iter_num)
                self.writer.add_scalar('Train/Loss_SoftIoU', float(loss_softiou.item()), self.iter_num)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.iter_num)

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(log_fmt.format(
                        epoch + 1, self.args.epochs,
                        self.iter_num % self.iter_per_epoch, self.iter_per_epoch,
                        self.optimizer.param_groups[0]['lr'],
                        loss_all.item(), loss_softiou.item(), loss_rec.item(),
                        cost_str, eta_str
                    ))

                step_val = self.args.save_iter_step > 0 and self.iter_num % self.args.save_iter_step == 0
                if step_val or self.iter_num % self.iter_per_epoch == 0:
                    self.validation()

    def validation(self):
        self.metric.reset()
        self.net.eval()
        for data, labels in self.val_data_loader:
            with torch.no_grad():
                out_D, out_T = self.net(data.to(self.device))
            self.metric.update(labels, out_T.cpu())

        iou, _, _, fmeasure = self.metric.get()
        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'latest.pkl'))
        if iou > self.best_iou:
            self.best_iou = iou
            torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'best.pkl'))
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure

        self.writer.add_scalar('Val/mIoU', iou, self.iter_num)
        self.writer.add_scalar('Val/F1', fmeasure, self.iter_num)
        self.logger.info(
            f'Validation [{self.args.dataset}] || mIoU: {iou * 100:.2f}% (Best: {self.best_iou * 100:.2f}%) || '
            f'F1: {fmeasure * 100:.2f}% (Best: {self.best_fmeasure * 100:.2f}%)'
        )


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.training()
    print(f'Training finished. Best mIoU: {trainer.best_iou * 100:.2f}%, Best F1: {trainer.best_fmeasure * 100:.2f}%')
