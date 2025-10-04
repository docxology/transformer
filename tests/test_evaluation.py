"""Tests for evaluation and benchmarking functionality."""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from src.transformer import Transformer, TransformerConfig
from src.evaluation import (
    compute_perplexity,
    compute_accuracy,
    evaluate_model,
    MetricsTracker,
    benchmark_inference,
    benchmark_training,
    profile_model
)


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        inputs = torch.randint(0, self.vocab_size, (self.seq_len,))
        targets = torch.randint(0, self.vocab_size, (self.seq_len,))
        return inputs, targets


@pytest.fixture
def small_model():
    """Create small model for testing."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.0
    )
    return Transformer(config)


class TestMetrics:
    """Test metric computation functions."""
    
    def test_compute_perplexity(self):
        """Test perplexity computation."""
        vocab_size = 100
        batch_size, seq_len = 2, 8
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        perplexity = compute_perplexity(logits, targets)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_compute_accuracy(self):
        """Test accuracy computation."""
        vocab_size = 10
        batch_size, seq_len = 2, 8
        
        # Create perfect predictions
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, targets[b, s]] = 10.0
        
        accuracy = compute_accuracy(logits, targets, top_k=1)
        
        assert accuracy == pytest.approx(1.0, abs=1e-6)
    
    def test_compute_accuracy_top_k(self):
        """Test top-k accuracy."""
        vocab_size = 10
        batch_size, seq_len = 2, 8
        
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Top-5 should be >= top-1
        top_1 = compute_accuracy(logits, targets, top_k=1)
        top_5 = compute_accuracy(logits, targets, top_k=5)
        
        assert top_5 >= top_1
    
    def test_accuracy_with_ignore_index(self):
        """Test accuracy with ignored indices."""
        vocab_size = 10
        batch_size, seq_len = 2, 8
        ignore_index = -100
        
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets[0, :4] = ignore_index  # Ignore first half of first sample
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        accuracy = compute_accuracy(logits, targets, ignore_index=ignore_index)
        
        assert 0 <= accuracy <= 1


class TestEvaluateModel:
    """Test full model evaluation."""
    
    def test_evaluate_model_basic(self, small_model):
        """Test basic model evaluation."""
        dataset = DummyDataset(
            vocab_size=small_model.config.vocab_size,
            seq_len=8,
            num_samples=10
        )
        dataloader = DataLoader(dataset, batch_size=2)
        device = torch.device('cpu')
        
        small_model.eval()
        results = evaluate_model(small_model, dataloader, device)
        
        assert results.num_samples == 10
        assert results.num_tokens > 0
        assert results.perplexity > 0
        assert 0 <= results.accuracy <= 1
        assert 0 <= results.top_5_accuracy <= 1
    
    def test_evaluation_results_to_dict(self, small_model):
        """Test results conversion to dict."""
        dataset = DummyDataset(
            vocab_size=small_model.config.vocab_size,
            seq_len=8,
            num_samples=5
        )
        dataloader = DataLoader(dataset, batch_size=2)
        device = torch.device('cpu')
        
        small_model.eval()
        results = evaluate_model(small_model, dataloader, device)
        
        results_dict = results.to_dict()
        
        assert isinstance(results_dict, dict)
        assert 'perplexity' in results_dict
        assert 'accuracy' in results_dict
        assert 'loss' in results_dict


class TestMetricsTracker:
    """Test metrics tracking."""
    
    def test_update_and_get_average(self):
        """Test updating and retrieving averages."""
        tracker = MetricsTracker()
        
        tracker.update(loss=1.0, accuracy=0.5)
        tracker.update(loss=2.0, accuracy=0.7)
        tracker.update(loss=3.0, accuracy=0.9)
        
        avg_loss = tracker.get_average('loss')
        avg_acc = tracker.get_average('accuracy')
        
        assert avg_loss == pytest.approx(2.0)
        assert avg_acc == pytest.approx(0.7)
    
    def test_get_all_averages(self):
        """Test getting all averages."""
        tracker = MetricsTracker()
        
        tracker.update(loss=1.0, accuracy=0.5)
        tracker.update(loss=2.0, accuracy=0.7)
        
        averages = tracker.get_all_averages()
        
        assert 'loss' in averages
        assert 'accuracy' in averages
        assert averages['loss'] == pytest.approx(1.5)
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = MetricsTracker()
        
        tracker.update(loss=1.0)
        tracker.reset()
        
        avg = tracker.get_average('loss')
        assert avg == 0.0


class TestBenchmarking:
    """Test benchmarking functions."""
    
    def test_benchmark_inference(self, small_model):
        """Test inference benchmarking."""
        device = torch.device('cpu')
        
        results = benchmark_inference(
            small_model,
            input_shape=(2, 8),
            vocab_size=small_model.config.vocab_size,
            device=device,
            num_warmup=2,
            num_runs=5,
            batch_sizes=[1, 2]
        )
        
        assert len(results) == 2
        assert 1 in results
        assert 2 in results
        
        for batch_size, result in results.items():
            assert result.mean_time > 0
            assert result.num_runs == 5
    
    def test_benchmark_training(self, small_model):
        """Test training benchmarking."""
        device = torch.device('cpu')
        optimizer = torch.optim.Adam(small_model.parameters())
        
        result = benchmark_training(
            small_model,
            input_shape=(2, 8),
            vocab_size=small_model.config.vocab_size,
            optimizer=optimizer,
            device=device,
            num_warmup=2,
            num_runs=5
        )
        
        assert result.mean_time > 0
        assert result.num_runs == 5
        assert result.throughput > 0
    
    def test_profile_model(self, small_model):
        """Test model profiling."""
        device = torch.device('cpu')
        
        profile = profile_model(
            small_model,
            input_shape=(2, 8),
            vocab_size=small_model.config.vocab_size,
            device=device
        )
        
        assert 'total_parameters' in profile
        assert 'trainable_parameters' in profile
        assert 'forward_time_ms' in profile
        assert 'approximate_flops' in profile
        assert profile['total_parameters'] > 0
        assert profile['forward_time_ms'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


