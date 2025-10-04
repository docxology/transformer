# Running the Transformer - Complete Guide

## Quick Start with `run.py`

The `run.py` script at the repository root provides a one-command solution to:
- Install and verify dependencies
- Run all examples
- Generate all visualizations and animations
- Create comprehensive outputs
- Validate the entire system

### Basic Usage

```bash
# Full setup and execution (recommended for first time)
python3 run.py

# With virtual environment (isolated dependencies)
python3 run.py --use-venv

# Skip dependency installation (if already installed)
python3 run.py --skip-install

# Only run examples (skip animations/validation)
python3 run.py --examples-only
```

### Using uv for Faster Installation

The script automatically detects and uses `uv` if available, providing 10-100x faster dependency installation:

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run - uv will be auto-detected
python3 run.py
```

**Benefits of uv:**
- 🚀 10-100x faster than pip
- 🔒 Better dependency resolution
- 💾 Efficient caching
- ✅ Compatible with pip/PyPI

## What `run.py` Does

### Phase 0: Environment Setup
- **Auto-detects uv** - Uses uv if available, falls back to pip
- **PYTHONPATH handling** - Automatically adds project root to Python path
- **Virtual environment** - Creates `.venv` if `--use-venv` flag is used

### Phase 1: Setup (Unless `--skip-install`)
1. **Dependency Check** - Verifies all required packages are installed
2. **Installation** - Installs missing dependencies (with uv or pip)
3. **Verification** - Runs `scripts/verify_installation.py` to test core functionality

### Phase 2: Examples
4. **Basic Usage** - Demonstrates core transformer functionality
5. **Path Analysis** - Analyzes information flow paths
6. **Animation Demo** - Creates visualization animations
7. **Training Example** - Shows complete training workflow

### Phase 3: Generation (Unless `--examples-only`)
8. **Generate Animations** - Creates all 9+ animation types
9. **Run Validation** - Comprehensive system validation
10. **Generate Documentation** - Builds complete documentation

## Expected Output

After running `python3 run.py`, you'll have:

```
outputs/
├── animations/          # 9+ GIF animations
│   ├── 01_information_flow.gif
│   ├── 02_attention_evolution.gif
│   ├── 03_path_trace.gif
│   └── ... (more animations)
│
├── visualizations/      # Static PNG plots
│   ├── attention_layer_0.png
│   ├── information_highways.png
│   └── ... (more visualizations)
│
├── analysis/           # JSON analysis reports
│   ├── comprehensive_analysis.json
│   ├── information_flow.json
│   └── model_profile.json
│
├── logs/               # Execution logs
│   └── session_*.log
│
└── reports/            # Generated reports
    ├── SYSTEM_REPORT.md
    └── COMPREHENSIVE_REPORT.md
```

## Execution Time

Typical execution times on standard hardware:

| Mode | Duration | What It Does |
|------|----------|--------------|
| `--examples-only` | ~2-5 min | Runs 4 examples |
| Full (default) | ~10-15 min | Everything |
| `--skip-install` | ~8-12 min | Skip dependency check |

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'src'

**Fixed automatically in the improved run.py!** The script now:
- Automatically adds project root to PYTHONPATH
- Passes PYTHONPATH to all subprocess calls
- Ensures imports work correctly

If you still encounter this:
```bash
# Set PYTHONPATH manually
export PYTHONPATH=/Users/4d/Documents/GitHub/transformer:$PYTHONPATH
python3 run.py
```

### Issue: Dependency Installation Fails

```bash
# Try with uv (faster and more reliable)
curl -LsSf https://astral.sh/uv/install.sh | sh
python3 run.py

# Or install dependencies manually
pip install -r requirements.txt

# Then run with skip flag
python3 run.py --skip-install

# Or use virtual environment
python3 run.py --use-venv
```

### Issue: Out of Memory

```bash
# Run examples only (lighter)
python3 run.py --examples-only

# Or modify examples to use smaller models
# Edit examples/*.py and use get_preset_config('tiny')
```

### Issue: Some Examples Fail

The script continues even if individual examples fail. Check:
- Error messages in terminal output
- Log files in `outputs/logs/`
- Ensure you're in the repository root

### Issue: Slow Execution

```bash
# Skip animation generation (time-consuming)
python3 run.py --examples-only

# Or run individual examples
python3 examples/basic_usage.py
```

## Manual Execution

If you prefer to run components individually:

```bash
# 1. Verify installation
python3 scripts/verify_installation.py

# 2. Run examples
python3 examples/basic_usage.py
python3 examples/path_analysis.py
python3 examples/animation_demo.py
python3 examples/training_example.py

# 3. Generate animations
python3 scripts/generate_all_animations.py

# 4. Run validation
python3 scripts/run_comprehensive_validation.py

# 5. Generate documentation
python3 scripts/generate_complete_documentation.py
```

## Advanced Usage

### Running in Background

```bash
# Run and log output
nohup python3 run.py > run.log 2>&1 &

# Check progress
tail -f run.log
```

### Running on Remote Server

```bash
# SSH into server
ssh your-server

# Clone and run
git clone <repository-url>
cd transformer
python3 run.py --skip-install  # if deps already installed
```

### Continuous Integration

```bash
# Add to CI pipeline
python3 run.py --examples-only  # Fast validation
```

## Output Management

### Viewing Animations

```bash
# Animations are in outputs/animations/
open outputs/animations/*.gif  # macOS
xdg-open outputs/animations/*.gif  # Linux
```

### Checking Reports

```bash
# View system report
cat outputs/reports/SYSTEM_REPORT.md

# View analysis
cat outputs/analysis/comprehensive_analysis.json | python3 -m json.tool
```

### Cleaning Outputs

```bash
# Remove all generated outputs
rm -rf outputs/

# Run will regenerate everything
python3 run.py
```

## Integration with Development

### Before Committing

```bash
# Verify everything works
python3 run.py --examples-only

# Check tests
pytest tests/ -v
```

### After Major Changes

```bash
# Full validation
python3 run.py
```

### Creating Release

```bash
# Generate all outputs
python3 run.py

# Ensure outputs directory is in .gitignore
# Commit code, not outputs
```

## Next Steps

After running `run.py` successfully:

1. **Explore Outputs** - Check `outputs/` directory
2. **Read Documentation** - Start with `doc/INDEX.md`
3. **Modify Examples** - Edit files in `examples/`
4. **Build Your Own** - Use as reference for your projects

## Related Documentation

- [Quick Start Guide](quickstart.md) - 5-minute manual tutorial
- [Examples README](../../examples/README.md) - Detailed example documentation
- [Visualization Guide](visualization.md) - Understanding outputs
- [Animation Guide](animations.md) - Animation reference

## See Also

- Repository structure: [README.md](../../README.md)
- Script documentation: [scripts/](../../scripts/)
- All documentation: [doc/INDEX.md](../INDEX.md)

---

**Pro Tip**: Run `python3 run.py --help` to see all available options!

