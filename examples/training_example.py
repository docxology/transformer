"""
Training example with visualization of learning dynamics.

Demonstrates:
1. Training loop setup
2. Loss monitoring
3. Visualization of attention pattern changes during training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.transformer import Transformer, TransformerConfig
from src.visualization import visualize_attention, plot_layer_statistics
from src.utils.output_paths import get_visualization_path
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleSequenceDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create input sequence
        seq = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Target is shifted by 1 (next token prediction)
        target = torch.cat([seq[1:], torch.randint(0, self.vocab_size, (1,))])
        return seq, target


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, model.config.vocab_size), targets.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, model.config.vocab_size), targets.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def visualize_training_state(model, sample_input, epoch, save_prefix):
    """Visualize model state during training."""
    model.eval()
    
    with torch.no_grad():
        logits, state = model(sample_input.unsqueeze(0), return_state=True)
    
    # Visualize attention patterns
    for layer_idx in range(min(2, model.config.n_layers)):
        attn_weights = state.get_attention_weights(layer_idx)
        path = get_visualization_path(f'{save_prefix}_epoch{epoch}_layer{layer_idx}.png', training=True)
        fig = visualize_attention(
            attn_weights,
            layer_idx=layer_idx,
            save_path=path
        )
        plt.close(fig)
    
    # Visualize layer statistics
    path = get_visualization_path(f'{save_prefix}_epoch{epoch}_stats.png', training=True)
    fig = plot_layer_statistics(state, save_path=path)
    plt.close(fig)
    
    model.train()


def main():
    print("=" * 60)
    print("Transformer Training Example")
    print("=" * 60)
    
    # Configuration
    config = TransformerConfig(
        vocab_size=200,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        dropout=0.1
    )
    
    # Training settings
    batch_size = 16
    seq_len = 32
    num_samples = 1000
    num_epochs = 10
    learning_rate = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    model = Transformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SimpleSequenceDataset(config.vocab_size, seq_len, num_samples)
    val_dataset = SimpleSequenceDataset(config.vocab_size, seq_len, num_samples // 5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Sample input for visualization
    sample_input = torch.randint(0, config.vocab_size, (seq_len,)).to(device)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Visualize every few epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Generating visualizations...")
            visualize_training_state(model, sample_input, epoch + 1, 'training')
    
    # Plot training curves
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', marker='o')
    ax.plot(val_losses, label='Val Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = get_visualization_path('training_curve.png', training=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {path}")
    
    # Final evaluation with full state analysis
    print("\nGenerating final state analysis...")
    model.eval()
    with torch.no_grad():
        logits, state = model(sample_input.unsqueeze(0), return_state=True)
    
    # Analyze attention patterns
    print("\nFinal Attention Pattern Analysis:")
    for layer_idx in range(config.n_layers):
        attn = state.get_attention_weights(layer_idx)
        sparsity = (attn[0].mean(0) > 0.1).float().mean().item()
        print(f"  Layer {layer_idx} sparsity: {sparsity:.3f}")
    
    print("\nTraining complete! Check outputs/ directory for visualizations.")


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)
    main()


