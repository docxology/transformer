"""
Generate complete documentation and reports for the transformer system.

Creates comprehensive reports covering all aspects of the system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.documentation import generate_documentation_reports
from src.utils.logging_system import TransformerLogger


def main():
    """Generate complete documentation."""

    logger = TransformerLogger("documentation_generator")

    logger.info("=" * 70)
    logger.info("COMPREHENSIVE DOCUMENTATION GENERATION")
    logger.info("=" * 70)

    # Delegate to the src/ entrypoint (manifest, counts, report rendering)
    logger.info("\nGenerating master index, file counts, and system report...")
    reports_dir = project_root / "outputs" / "reports"
    result = generate_documentation_reports(project_root, reports_dir)

    for path in result["paths"]:
        logger.info(f"✓ Saved: {path}")

    counts = result["counts"]
    logger.info("\n" + "=" * 70)
    logger.info("DOCUMENTATION GENERATION COMPLETE")
    logger.info("=" * 70)

    logger.info(f"\nGenerated files:")
    for path in result["paths"]:
        logger.info(f"  • {path.name}")

    logger.info(f"\nTotal project files: {counts['total']}")
    logger.info(f"Documentation files: {counts['doc_files']}")
    logger.info(f"Output files: {counts['animation_files'] + counts['visualization_files'] + counts['analysis_files']}")

    logger.save_metrics()
    logger.info("\n✅ DOCUMENTATION COMPLETE")


if __name__ == '__main__':
    main()
