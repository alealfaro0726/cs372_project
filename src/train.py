
import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from model import ConditionalTransformer
from midi_tokenizer import MIDITokenizer


class MIDIDataset(Dataset):

    def __init__(self, data: List[Dict], max_length: int = 512, seed: int = 42):
        self.data = data
        self.max_length = max_length
        self.seed = seed

        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]['tokens'][:self.max_length]

        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)

        rng_state = np.random.RandomState(self.seed + idx)

        image_embed = torch.tensor(rng_state.randn(512).astype(np.float32))

        emotion_label = idx % 5

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'image_embed': image_embed,
            'emotion_label': emotion_label
        }


def collate_fn(batch):
    return {
        'input_tokens': torch.stack([item['input_tokens'] for item in batch]),
        'target_tokens': torch.stack([item['target_tokens'] for item in batch]),
        'image_embeds': torch.stack([item['image_embed'] for item in batch]),
        'emotion_labels': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long)
    }


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str,
        output_dir: Path
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.scheduler = self._create_scheduler()

        self.use_amp = config['training'].get('mixed_precision', False) and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def _create_scheduler(self):
        scheduler_type = self.config['training'].get('lr_scheduler', 'cosine')

        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3
            )
        else:
            return None

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch in pbar:
            input_tokens = batch['input_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)
            image_embeds = batch['image_embeds'].to(self.device)
            emotion_labels = batch['emotion_labels'].to(self.device)

            if self.use_amp:
                with autocast():
                    _, loss = self.model(input_tokens, image_embeds, emotion_labels, target_tokens)
            else:
                _, loss = self.model(input_tokens, image_embeds, emotion_labels, target_tokens)

            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('grad_clip', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('grad_clip', 1.0)
                )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            input_tokens = batch['input_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)
            image_embeds = batch['image_embeds'].to(self.device)
            emotion_labels = batch['emotion_labels'].to(self.device)

            _, loss = self.model(input_tokens, image_embeds, emotion_labels, target_tokens)

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print("Warning: Validation set is empty!")
            return float('inf')

        return total_loss / num_batches

    def save_checkpoint(self, filename: str, is_best: bool = False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }

        checkpoint_path = self.output_dir / 'models' / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.output_dir / 'models' / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history['learning_rates'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)

        plt.tight_layout()
        plot_path = self.output_dir / 'logs' / 'training_curves.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved training curves: {plot_path}")

    def train(self):
        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training'].get('early_stopping_patience', 10)

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")

        for epoch in range(epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()

            val_loss = self.validate()

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % self.config['training'].get('save_every', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', is_best)

            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

        self.save_checkpoint('final_model.pt')
        self.plot_training_curves()

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def load_config(config_path: Optional[str] = None) -> Dict:
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return {
            'model': {
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'd_ff': 2048,
                'dropout': 0.1,
                'max_seq_len': 512,
                'image_embed_dim': 512,
                'emotion_embed_dim': 64
            },
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'weight_decay': 0.01,
                'warmup_steps': 1000,
                'lr_scheduler': 'cosine',
                'grad_clip': 1.0,
                'mixed_precision': True,
                'save_every': 5,
                'eval_every': 1,
                'early_stopping_patience': 10
            },
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Train Conditional Transformer')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for models and logs')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)

    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    print(f"Loading data from {data_dir}...")

    with open(data_dir / 'train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / 'val.pkl', 'rb') as f:
        val_data = pickle.load(f)

    print(f"Train sequences: {len(train_data)}")
    print(f"Val sequences: {len(val_data)}")

    tokenizer = MIDITokenizer.load_vocab(str(data_dir / 'vocab.json'))
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    train_dataset = MIDIDataset(train_data, max_length=config['model']['max_seq_len'], seed=args.seed)
    val_dataset = MIDIDataset(val_data, max_length=config['model']['max_seq_len'], seed=args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print("Creating model...")
    model = ConditionalTransformer(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        image_embed_dim=config['model']['image_embed_dim'],
        n_emotions=5,
        emotion_embed_dim=config['model']['emotion_embed_dim']
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    output_dir = Path(args.output_dir)
    trainer = Trainer(model, train_loader, val_loader, config, device, output_dir)

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    trainer.train()


if __name__ == '__main__':
    main()
