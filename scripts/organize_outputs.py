"""
Organize outputs directory structure.

Moves misplaced files to correct subdirectories and ensures
proper organization of all output files.
"""

import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced_logging import create_enhanced_logger


def organize_outputs(outputs_dir: Path, dry_run: bool = False):
    """
    Organize outputs directory by moving files to appropriate subdirectories.
    
    Args:
        outputs_dir: Path to outputs directory
        dry_run: If True, only show what would be done without moving files
    """
    logger = create_enhanced_logger()
    
    logger.header("Output Directory Organization", "=", "OKBLUE")
    
    # Define organization rules
    rules = {
        'animations': {
            'patterns': ['*.gif'],
            'exceptions': []
        },
        'visualizations': {
            'patterns': ['*.png'],
            'exceptions': []
        },
        'analysis': {
            'patterns': ['*_flow.json', '*_profile.json', '*_analysis.json'],
            'exceptions': []
        },
        'data': {
            'patterns': ['*.pt', '*.pth', '*.h5'],
            'exceptions': []
        },
        'configs': {
            'patterns': ['*config*.json', '*config*.yaml'],
            'exceptions': []
        },
        'reports': {
            'patterns': ['*.md', 'summary_*.txt'],
            'exceptions': []
        },
        'logs': {
            'patterns': ['*.log', 'metrics_*.json', 'session_*.json'],
            'exceptions': []
        }
    }
    
    # Create training subdirectory in visualizations
    training_viz_dir = outputs_dir / 'visualizations' / 'training'
    training_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files in root outputs directory
    root_files = [f for f in outputs_dir.iterdir() if f.is_file()]
    
    if not root_files:
        logger.success("Outputs directory is already organized")
        return
    
    logger.section(f"Found {len(root_files)} files in root directory")
    
    moves = []
    
    for file_path in root_files:
        moved = False
        
        # Special handling for training visualizations
        if file_path.name.startswith('training_') and file_path.suffix == '.png':
            dest_dir = training_viz_dir
            dest_path = dest_dir / file_path.name
            moves.append((file_path, dest_path, 'visualizations/training'))
            moved = True
            continue
        
        # Check against rules
        for category, rule in rules.items():
            if moved:
                break
            
            for pattern in rule['patterns']:
                if file_path.match(pattern):
                    # Check if it's an exception
                    is_exception = any(file_path.match(exc) for exc in rule['exceptions'])
                    
                    if not is_exception:
                        dest_dir = outputs_dir / category
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / file_path.name
                        
                        # Check if file already exists in destination
                        if dest_path.exists():
                            # If files are different, create a backup
                            if dest_path.stat().st_size != file_path.stat().st_size:
                                backup_path = dest_path.with_suffix(dest_path.suffix + '.backup')
                                moves.append((dest_path, backup_path, f'{category} (backup)'))
                        
                        moves.append((file_path, dest_path, category))
                        moved = True
                        break
        
        if not moved:
            logger.warning(f"No rule for: {file_path.name}", "Consider adding to rules")
    
    # Display move plan
    logger.section("Move Plan")
    
    categories = {}
    for src, dest, category in moves:
        if category not in categories:
            categories[category] = []
        categories[category].append((src, dest))
    
    for category, files in sorted(categories.items()):
        logger.info(f"{category}: {len(files)} files", indent=0)
        for src, dest in files[:5]:  # Show first 5
            logger.detail(src.name, f"→ {dest.relative_to(outputs_dir)}", indent=2)
        if len(files) > 5:
            logger.detail("...", f"and {len(files) - 5} more", indent=2)
    
    # Execute moves
    if dry_run:
        logger.warning("Dry run - no files moved")
        return
    
    logger.section("Executing Moves")
    
    errors = []
    success_count = 0
    
    for src, dest, category in moves:
        try:
            # Create parent directory if needed
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(src), str(dest))
            success_count += 1
            
        except Exception as e:
            error_msg = f"Failed to move {src.name}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    # Summary
    logger.section("Summary")
    logger.detail("Files moved", success_count, indent=1)
    logger.detail("Errors", len(errors), indent=1)
    
    if errors:
        logger.section("Errors")
        for error in errors:
            logger.list_item(error, status="error")
    else:
        logger.success(f"Successfully organized {success_count} files")
    
    # Show final structure
    logger.section("Final Directory Structure")
    show_directory_structure(outputs_dir, logger)


def show_directory_structure(outputs_dir: Path, logger):
    """Display organized directory structure."""
    subdirs = ['animations', 'visualizations', 'analysis', 'data', 
                'configs', 'reports', 'logs']
    
    total_files = 0
    total_size = 0
    
    for subdir in subdirs:
        subdir_path = outputs_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            size_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
            
            total_files += file_count
            total_size += size_mb
            
            logger.detail(
                f"{subdir:20s}",
                f"{file_count:3d} files ({size_mb:6.2f} MB)",
                indent=1
            )
            
            # Show subdirectories
            subdirs_in = [d for d in subdir_path.iterdir() if d.is_dir()]
            for sub in subdirs_in:
                sub_files = len([f for f in sub.rglob('*') if f.is_file()])
                sub_size = sum(f.stat().st_size for f in sub.rglob('*') if f.is_file()) / 1024 / 1024
                logger.detail(
                    f"  └─ {sub.name:17s}",
                    f"{sub_files:3d} files ({sub_size:6.2f} MB)",
                    indent=1
                )
    
    logger.detail("─" * 40, "", indent=1)
    logger.detail(
        f"{'TOTAL':20s}",
        f"{total_files:3d} files ({total_size:6.2f} MB)",
        indent=1
    )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize outputs directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually moving files'
    )
    
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=project_root / 'outputs',
        help='Path to outputs directory (default: outputs/)'
    )
    
    args = parser.parse_args()
    
    # Ensure outputs directory exists
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize
    organize_outputs(args.outputs_dir, dry_run=args.dry_run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

