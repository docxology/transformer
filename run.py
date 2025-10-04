#!/usr/bin/env python3
"""
First-Principles Transformer - Complete Setup and Execution Script

This script:
1. Sets up Python environment (with uv if available)
2. Verifies and installs dependencies
3. Runs installation verification
4. Executes all examples
5. Generates all visualizations and outputs

Usage:
    python3 run.py                    # Run everything
    python3 run.py --skip-install     # Skip dependency check
    python3 run.py --examples-only    # Only run examples
    python3 run.py --use-venv         # Create and use virtual environment
"""

import sys
import subprocess
import os
import time
import argparse
import shutil
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.utils.enhanced_logging import create_enhanced_logger


class TransformerRunner:
    """Orchestrates complete transformer setup and execution."""
    
    def __init__(self, skip_install=False, examples_only=False, use_venv=False):
        self.skip_install = skip_install
        self.examples_only = examples_only
        self.use_venv = use_venv
        self.root_dir = Path(__file__).parent.resolve()
        self.start_time = time.time()
        self.venv_python = None
        self.has_uv = self._check_uv()
        self.logger = create_enhanced_logger()
        
        # Ensure project root is in Python path
        if str(self.root_dir) not in sys.path:
            sys.path.insert(0, str(self.root_dir))
        
    def _check_uv(self):
        """Check if uv is installed."""
        return shutil.which("uv") is not None
        
    def print_header(self, message, char="="):
        """Print a formatted header (using enhanced logger)."""
        self.logger.header(message, char)
        
    def print_step(self, step, total, message):
        """Print a step indicator (using enhanced logger)."""
        self.logger.step(step, total, message)
        
    def run_command(self, command, description, capture_output=False, env_vars=None):
        """Run a shell command with error handling."""
        cmd_str = ' '.join(str(c) for c in command)
        self.logger.command(cmd_str, description)
        
        # Set up environment
        env = os.environ.copy()
        
        # Always add project root to PYTHONPATH
        pythonpath = str(self.root_dir)
        if 'PYTHONPATH' in env:
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env['PYTHONPATH'] = pythonpath
        
        # Add custom env vars
        if env_vars:
            env.update(env_vars)
        
        try:
            result = subprocess.run(
                command,
                cwd=self.root_dir,
                check=True,
                capture_output=capture_output,
                text=True,
                env=env
            )
            self.logger.step_summary(description)
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{description} failed with error code {e.returncode}")
            if capture_output and e.stderr:
                self.logger.detail("Error", e.stderr, indent=1)
            return None
    
    def get_python_executable(self):
        """Get the Python executable to use."""
        if self.venv_python and self.venv_python.exists():
            return str(self.venv_python)
        return sys.executable
    
    def setup_virtual_environment(self):
        """Set up a virtual environment using uv or venv."""
        if not self.use_venv:
            return True
            
        self.print_step(1, 8, "Setting up virtual environment")
        
        venv_dir = self.root_dir / ".venv"
        
        if venv_dir.exists():
            self.logger.success("Virtual environment already exists", str(venv_dir))
        else:
            if self.has_uv:
                self.logger.info("Using uv to create virtual environment", indent=1)
                result = self.run_command(
                    ["uv", "venv", ".venv"],
                    "Creating virtual environment with uv"
                )
            else:
                self.logger.info("Using venv to create virtual environment", indent=1)
                result = self.run_command(
                    [sys.executable, "-m", "venv", ".venv"],
                    "Creating virtual environment with venv"
                )
            
            if not result:
                self.logger.error("Failed to create virtual environment")
                return False
        
        # Determine Python executable in venv
        if os.name == 'nt':  # Windows
            self.venv_python = venv_dir / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            self.venv_python = venv_dir / "bin" / "python"
        
        if not self.venv_python.exists():
            self.logger.error("Virtual environment Python not found", str(self.venv_python))
            return False
            
        self.logger.success("Using Python", str(self.venv_python))
        return True
            
    def check_dependencies(self):
        """Check if required packages are installed."""
        step = 2 if self.use_venv else 1
        total = 8 if self.use_venv else 7
        self.print_step(step, total, "Checking dependencies")
        
        python_exec = self.get_python_executable()
        
        check_script = """
import sys
try:
    import torch
    import numpy
    import matplotlib
    import seaborn
    import plotly
    import pytest
    import yaml
    import networkx
    print("SUCCESS")
except ImportError as e:
    print(f"MISSING: {e.name}")
    sys.exit(1)
"""
        
        try:
            result = subprocess.run(
                [python_exec, "-c", check_script],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                self.logger.success("All required packages are installed")
                return True
            else:
                self.logger.error("Missing packages detected")
                if result.stdout:
                    self.logger.detail("Missing", result.stdout.strip(), indent=1)
                return False
        except Exception as e:
            self.logger.error("Dependency check failed", str(e))
            return False
            
    def install_dependencies(self):
        """Install dependencies using uv or pip."""
        step = 3 if self.use_venv else 2
        total = 8 if self.use_venv else 7
        self.print_step(step, total, "Installing dependencies")
        
        requirements_file = self.root_dir / "requirements.txt"
        if not requirements_file.exists():
            self.logger.error("requirements.txt not found")
            return False
        
        python_exec = self.get_python_executable()
        
        if self.has_uv:
            self.logger.info("Using uv for fast installation", indent=1)
            result = self.run_command(
                ["uv", "pip", "install", "-r", "requirements.txt"],
                "Installing packages with uv"
            )
        else:
            self.logger.info("Using pip for installation", indent=1)
            result = self.run_command(
                [python_exec, "-m", "pip", "install", "-r", "requirements.txt"],
                "Installing packages with pip"
            )
        
        return result is not None
        
    def verify_installation(self):
        """Run the installation verification script."""
        step = 4 if self.use_venv else 3
        total = 8 if self.use_venv else 7
        self.print_step(step, total, "Verifying installation")
        
        verify_script = self.root_dir / "scripts" / "verify_installation.py"
        if not verify_script.exists():
            self.logger.error("Verification script not found")
            return False
        
        python_exec = self.get_python_executable()
        result = self.run_command(
            [python_exec, str(verify_script)],
            "Running verification tests"
        )
        return result is not None
        
    def run_example(self, example_name):
        """Run a single example script."""
        example_path = self.root_dir / "examples" / f"{example_name}.py"
        
        if not example_path.exists():
            self.logger.error(f"Example not found: {example_name}")
            return False
            
        self.logger.subheader(f"Running: {example_name}")
        
        python_exec = self.get_python_executable()
        result = self.run_command(
            [python_exec, str(example_path)],
            f"Executing {example_name}"
        )
        return result is not None
        
    def run_all_examples(self):
        """Run all example scripts in order."""
        step = 5 if self.use_venv else 4
        total = 8 if self.use_venv else 7
        self.print_step(step, total, "Running examples")
        
        examples = [
            "basic_usage",
            "path_analysis",
            "animation_demo",
            "training_example"
        ]
        
        results = {}
        for idx, example in enumerate(examples, 1):
            self.logger.info(f"Example {idx}/{len(examples)}: {example}", indent=0)
            results[example] = self.run_example(example)
            
        # Summary
        self.logger.header("Examples Summary", "=")
        for example, success in results.items():
            status = "success" if success else "error"
            self.logger.list_item(example, status=status)
        
        return all(results.values())
        
    def generate_animations(self):
        """Generate all animations."""
        step = 6 if self.use_venv else 5
        total = 8 if self.use_venv else 7
        self.print_step(step, total, "Generating animations")
        
        anim_script = self.root_dir / "scripts" / "generate_all_animations.py"
        if not anim_script.exists():
            self.logger.error("Animation generation script not found")
            return False
        
        python_exec = self.get_python_executable()
        result = self.run_command(
            [python_exec, str(anim_script)],
            "Creating animations"
        )
        return result is not None
    
    def generate_architecture_viz(self):
        """Generate architecture visualizations."""
        step = 6 if self.use_venv else 5
        total = 9 if self.use_venv else 8  # Updated total
        self.print_step(step, total, "Generating architecture visualizations")
        
        arch_script = self.root_dir / "scripts" / "generate_architecture_visualizations.py"
        if not arch_script.exists():
            self.logger.error("Architecture visualization script not found")
            return False
        
        python_exec = self.get_python_executable()
        result = self.run_command(
            [python_exec, str(arch_script)],
            "Creating architecture diagrams"
        )
        return result is not None
        
    def run_validation(self):
        """Run comprehensive validation."""
        step = 8 if self.use_venv else 7
        total = 9 if self.use_venv else 8
        self.print_step(step, total, "Running comprehensive validation")
        
        validation_script = self.root_dir / "scripts" / "run_comprehensive_validation.py"
        if not validation_script.exists():
            self.logger.error("Validation script not found")
            return False
        
        python_exec = self.get_python_executable()
        result = self.run_command(
            [python_exec, str(validation_script)],
            "Running validation suite"
        )
        return result is not None
        
    def generate_documentation(self):
        """Generate complete documentation."""
        step = 9 if self.use_venv else 8
        total = 9 if self.use_venv else 8
        self.print_step(step, total, "Generating documentation")
        
        doc_script = self.root_dir / "scripts" / "generate_complete_documentation.py"
        if not doc_script.exists():
            self.logger.error("Documentation generation script not found")
            return False
        
        python_exec = self.get_python_executable()
        result = self.run_command(
            [python_exec, str(doc_script)],
            "Building documentation"
        )
        return result is not None
        
    def show_outputs(self):
        """Display information about generated outputs."""
        self.logger.header("Generated Outputs", "=")
        
        outputs_dir = self.root_dir / "outputs"
        if not outputs_dir.exists():
            self.logger.warning("No outputs directory found")
            return
            
        # Count files in each subdirectory
        subdirs = {
            "animations": "Animations (GIF files)",
            "visualizations": "Static visualizations (PNG files)",
            "analysis": "Analysis reports (JSON files)",
            "logs": "Execution logs",
            "reports": "Generated reports",
            "data": "Generated data files",
            "configs": "Saved configurations"
        }
        
        self.logger.section("Output Directory Structure")
        total_files = 0
        total_size = 0
        
        for subdir, description in subdirs.items():
            subdir_path = outputs_dir / subdir
            if subdir_path.exists():
                files = [f for f in subdir_path.iterdir() if f.is_file()]
                count = len(files)
                size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
                total_files += count
                total_size += size_mb
                
                self.logger.detail(
                    subdir,
                    f"{count:2d} files ({size_mb:.2f} MB) - {description}",
                    indent=1
                )
            else:
                self.logger.list_item(f"{subdir} (not created)", status="pending")
        
        self.logger.detail("Output Directory", outputs_dir, indent=0)
        self.logger.detail("Total Files", total_files, indent=0)
        self.logger.detail("Total Size", f"{total_size:.2f} MB", indent=0)
        
    def show_environment_info(self):
        """Display environment information."""
        env_info = {
            "Python": sys.version.split()[0],
            "Executable": self.get_python_executable(),
            "Project root": str(self.root_dir),
            "uv": "Available ✓" if self.has_uv else "Not installed (using pip)"
        }
        
        if self.use_venv:
            env_info["Virtual env"] = str(self.root_dir / '.venv')
        
        self.logger.system_info(env_info)
        
    def print_summary(self):
        """Print execution summary."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        self.logger.header("✨ Execution Complete ✨", "=", "OKGREEN")
        
        summary_items = {
            "Total Time": f"{minutes}m {seconds}s",
            "Status": "Success"
        }
        
        self.logger.summary_box(summary_items)
        
        self.logger.section("Next Steps")
        next_steps = [
            "View outputs in the 'outputs/' directory",
            "Check animations in 'outputs/animations/'",
            "Read generated reports in 'outputs/reports/'",
            "See documentation in 'doc/'"
        ]
        for step in next_steps:
            self.logger.list_item(step)
        
        self.logger.section("Documentation")
        docs = [
            "README.md - Project overview",
            "doc/INDEX.md - Documentation index",
            "doc/guides/quickstart.md - Quick start guide"
        ]
        for doc in docs:
            self.logger.list_item(doc)
        
        if self.use_venv:
            self.logger.section("Virtual Environment")
            if os.name == 'nt':
                self.logger.detail("Activate", ".venv\\Scripts\\activate", indent=1)
            else:
                self.logger.detail("Activate", "source .venv/bin/activate", indent=1)
        
    def run(self):
        """Execute the complete workflow."""
        self.logger.header("First-Principles Transformer - Complete Setup & Execution", "=", "HEADER")
        
        self.show_environment_info()
        
        self.logger.section("Execution Plan")
        steps = []
        if self.use_venv:
            steps.append("Set up virtual environment")
        if not self.skip_install:
            steps.extend([
                "Check and install dependencies (with uv if available)",
                "Verify installation"
            ])
        if not self.examples_only:
            steps.extend([
                "Run all examples",
                "Generate animations",
                "Generate architecture visualizations",
                "Run validation",
                "Generate documentation"
            ])
        else:
            steps.append("Run all examples only")
        
        for i, step in enumerate(steps, 1):
            self.logger.list_item(f"{i}. {step}")
        
        success = True
        
        try:
            # Virtual environment setup
            if self.use_venv:
                if not self.setup_virtual_environment():
                    print("\n✗ Failed to set up virtual environment")
                    return False
            
            # Dependencies
            if not self.skip_install and not self.examples_only:
                if not self.check_dependencies():
                    if not self.install_dependencies():
                        self.logger.error("Failed to install dependencies")
                        return False
                    # Re-check after installation
                    if not self.check_dependencies():
                        self.logger.error("Dependencies still missing after installation")
                        return False
                        
                if not self.verify_installation():
                    self.logger.error("Installation verification failed")
                    return False
                    
            # Examples
            if not self.run_all_examples():
                self.logger.warning("Some examples failed, but continuing...")
                success = False
                
            # Generation and validation
            if not self.examples_only:
                if not self.generate_animations():
                    self.logger.warning("Animation generation had issues, but continuing...")
                    success = False
                
                if not self.generate_architecture_viz():
                    self.logger.warning("Architecture visualization had issues, but continuing...")
                    success = False
                    
                if not self.run_validation():
                    self.logger.warning("Validation had issues, but continuing...")
                    success = False
                    
                if not self.generate_documentation():
                    self.logger.warning("Documentation generation had issues, but continuing...")
                    success = False
                    
            # Show outputs
            self.show_outputs()
            
            # Summary
            self.print_summary()
            
            return success
            
        except KeyboardInterrupt:
            self.logger.error("\n\nExecution interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="First-Principles Transformer - Complete Setup & Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run.py                   # Full setup and execution
  python3 run.py --skip-install    # Skip dependency installation
  python3 run.py --examples-only   # Only run examples
  python3 run.py --use-venv        # Use virtual environment
  
Environment:
  This script automatically uses 'uv' if available for faster installation.
  Otherwise, it falls back to 'pip'.
  
  To install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
        """
    )
    
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency check and installation"
    )
    
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Only run examples, skip other generation steps"
    )
    
    parser.add_argument(
        "--use-venv",
        action="store_true",
        help="Create and use a virtual environment (.venv)"
    )
    
    args = parser.parse_args()
    
    runner = TransformerRunner(
        skip_install=args.skip_install,
        examples_only=args.examples_only,
        use_venv=args.use_venv
    )
    
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
