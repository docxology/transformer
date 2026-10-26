# Frequently Asked Questions

## General Questions

### Q: What makes this transformer implementation different?

**A:** This implementation prioritizes **understanding and observability**:
- Full state tracking of residual and K/V streams
- Comprehensive visualization tools
- First-principles approach with detailed documentation
- Modular, testable, extensible design
- Educational focus with extensive comments

### Q: Is this production-ready?

**A:** It's designed for **research, education, and prototyping**:
- ✅ Correct implementation with full test coverage
- ✅ Modular and maintainable code
- ✅ Complete visualization capabilities
- ❌ Not optimized for production speed
- ❌ Lacks production features (distributed training, quantization, etc.)

For production, consider PyTorch's built-in transformers or HuggingFace Transformers.

### Q: Can I use this for actual training?

**A:** Yes! The implementation is correct and functional:
- All components are fully working
- Gradient flow is correct
- Can train on real data
- Just slower than optimized implementations

## Architecture Questions

### Q: What are the "two information highways"?

**A:**
1. **Residual Stream (Vertical)**: Information flows UP through layers at each position
2. **K/V Stream (Horizontal)**: Information flows RIGHT across positions via attention

See [Information Flow](concepts/information_flow.md) for details.

### Q: What is the "path explosion"?

**A:** The astronomical number of computational paths through the network. For n layers and m positions, there are C(n+m, n) distinct paths. For a 6-layer, 32-position model, that's ~2.3 million paths!

This redundancy enables:
- Robust information transmission
- Rich interference patterns
- Nuanced encodings

See [Path Explosion](theory/TRANSFORMER_THEORY.md#computational-path-explosion).

### Q: Can transformers really introspect?

**A:** Yes! The architecture explicitly enables it:
- K/V stream provides access to past position states
- Residual stream preserves past layer states
- Both are accessible via attention and residual connections

This contradicts some common misconceptions.

## Implementation Questions

### Q: Why is state tracking optional?

**A:** State tracking adds memory overhead:
- Stores intermediate activations
- Keeps attention weights
- Maintains full computational history

Use `return_state=True` when:
- Visualizing
- Analyzing
- Debugging

Use `return_state=False` when:
- Training (saves memory)
- Production inference
- Speed is critical

### Q: How do I make it faster?

**A:**
1. **Disable state tracking**: `return_state=False`
2. **Use torch.compile**: `model = torch.compile(model)`
3. **Enable CUDA**: `model.to('cuda')`
4. **Reduce batch size**: If memory constrained
5. **Use smaller models**: Start with 'small' preset
6. **Implement Flash Attention**: For long sequences

### Q: Why use pre-norm instead of post-norm?

**A:** Pre-norm (LayerNorm before attention/FFN) provides:
- More stable training
- Better gradient flow
- Modern best practice

Post-norm (original paper) is less stable for deep models.

### Q: What's the difference between d_model and d_ff?

**A:**
- **d_model**: Model dimension (residual stream width). Constant through network.
- **d_ff**: Feed-forward dimension. Typically 4×d_model. Internal to FFN, expands then contracts.

## Usage Questions

### Q: How do I choose model size?

**A:** Use presets as starting points:
- **debug** (64-dim, 2-layer): Quick testing
- **tiny** (128-dim, 2-layer): Small experiments
- **small** (256-dim, 4-layer): Most experiments
- **base** (512-dim, 6-layer): Serious training
- **large** (1024-dim, 12-layer): Large-scale

Rule of thumb: Start small, scale up only if needed.

### Q: What sequence length should I use?

**A:** Depends on task and memory:
- **32-128**: Short sequences, limited memory
- **256-512**: Most tasks
- **1024+**: Long documents (requires lots of memory)

Attention is O(n²) in sequence length!

### Q: How do I debug training issues?

**A:**
1. **Visualize gradients**: `visualize_gradient_flow(model)`
2. **Check attention**: Are patterns reasonable?
3. **Monitor norms**: `plot_layer_statistics(state)`
4. **Reduce learning rate**: Often solves instability
5. **Check data**: Ensure inputs are sensible
6. **Overfit single batch**: Sanity check

## Visualization Questions

### Q: Why are my visualizations blank/white?

**A:** Common causes:
- Forgot `return_state=True`
- Model not on CPU (some visualizations need CPU tensors)
- Empty/zero attention weights (check model trained)
- Wrong indices (layer/position out of range)

### Q: How do I visualize during training?

**A:** Use `TrainingVisualizer`:
```python
viz = TrainingVisualizer()
for epoch in range(num_epochs):
    loss = train_epoch(...)
    viz.log_metrics(train_loss=loss)
    
fig = viz.plot_training_curves()
```

### Q: Can I make interactive visualizations?

**A:** Yes! Use `plot_attention_heads` for Plotly interactive plots:
```python
fig = plot_attention_heads(attn_weights)
fig.show()  # Opens in browser
```

## Performance Questions

### Q: Why is it slow compared to HuggingFace?

**A:** This implementation prioritizes clarity:
- Full state tracking
- No fused kernels
- No low-level optimizations
- Modular over fast

For production speed, use optimized libraries.

### Q: How much memory does it need?

**A:** Approximate for base model (512-dim, 6-layer):
- **Parameters**: ~40M → ~160MB (fp32)
- **Activations**: ~batch_size × seq_len × d_model × n_layers
- **State tracking**: Adds ~2× overhead

### Q: Can I use mixed precision?

**A:** Yes, but requires modification:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, state = model(inputs)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Configuration Questions

### Q: How do I save/load configurations?

**A:**
```python
from src.config import save_config, load_config

# Save
save_config(config, 'my_config.json')

# Load
config = load_config('my_config.json')
```

Supports JSON and YAML.

### Q: Can I modify presets?

**A:** Yes, copy and modify:
```python
from src.config import get_preset_config

config = get_preset_config('base')
config.n_layers = 8  # Modify
config.dropout = 0.2
```

## Testing Questions

### Q: How do I run tests?

**A:**
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_components.py -v

# Specific test
pytest tests/test_model.py::TestTransformer::test_forward_shape -v
```

### Q: How do I add tests?

**A:** Follow existing patterns in `tests/`:
1. Create test class
2. Write test methods (start with `test_`)
3. Use fixtures for common setup
4. Run and verify

See [Testing Guide](guides/IMPLEMENTATION_GUIDE.md#testing).

## Common Issues

### Q: ImportError: cannot import name 'X'

**A:** Ensure you're in project root:
```bash
cd /path/to/transformer
python3 -c "from src.transformer import Transformer"
```

### Q: CUDA out of memory

**A:**
1. Reduce batch size
2. Use smaller model
3. Reduce sequence length
4. Disable state tracking
5. Use gradient checkpointing (requires modification)

### Q: Gradients are NaN

**A:**
1. Reduce learning rate
2. Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Check for inf/nan in data
4. Use mixed precision carefully

### Q: Attention weights are uniform

**A:** This can be normal:
- Early in training
- When aggregating information (not focusing)
- For some heads

Check if loss is decreasing. Uniform attention isn't always bad.

## Advanced Questions

### Q: Can I add custom attention patterns?

**A:** Yes, modify `MultiHeadAttention`:
1. Subclass `MultiHeadAttention`
2. Override `forward` method
3. Implement custom attention computation
4. Maintain state tracking interface

### Q: How do I implement custom layers?

**A:** Follow the component pattern:
1. Inherit from `nn.Module`
2. Implement `__init__` and `forward`
3. Return state if needed
4. Add tests
5. Update documentation

### Q: Can I use this for non-autoregressive models?

**A:** Yes, just remove causal masking:
```python
# Don't pass mask, or pass bidirectional mask
logits, state = model(input_ids, mask=None)
```

## Contributing Questions

### Q: How do I contribute?

**A:** See [Contributing Guide](../CONTRIBUTING.md):
1. Fork repository
2. Create feature branch
3. Write code + tests + docs
4. Submit pull request

### Q: What kind of contributions are welcome?

**A:**
- Bug fixes
- New visualizations
- Performance improvements
- Documentation improvements
- Example notebooks
- Test coverage
- Educational content

## Still Have Questions?

- Check [Documentation Index](INDEX.md)
- Read [Troubleshooting Guide](guides/running.md#troubleshooting)
- Look at [Examples](../examples/)
- Open an issue on GitHub


