"""Tests for the documentation manifest and report module."""

from pathlib import Path

from src.utils.documentation import (
    DOC_STRUCTURE,
    build_master_index,
    count_project_files,
    count_test_functions,
    generate_documentation_reports,
)

PROJECT_ROOT = Path(__file__).parent.parent


class TestDocumentationManifest:
    """Test the documentation manifest stays in sync with the repo."""

    def test_all_manifest_paths_exist(self):
        """Every documented file must exist on disk (staleness guard)."""
        missing = [
            path
            for group in DOC_STRUCTURE.values()
            for path in group
            if not (PROJECT_ROOT / path).exists()
        ]
        assert missing == []

    def test_master_index_is_deterministic(self):
        """Repeated builds produce identical output; no wall-clock stamps."""
        index1 = build_master_index(PROJECT_ROOT)
        index2 = build_master_index(PROJECT_ROOT)

        assert index1 == index2
        assert 'generated' not in index1

    def test_capabilities_derive_from_repo(self):
        """Derivable capability counts match the live repo."""
        index = build_master_index(PROJECT_ROOT)

        assert index['capabilities']['tests'] > 0
        assert index['capabilities']['config_presets'] == 7
        assert index['capabilities']['documentation_files'] > 0
        assert index['capabilities']['total_python_files'] > 0


class TestFileCounts:
    """Test project file counting."""

    def test_counts_are_positive_integers(self):
        """All categories count a non-negative number of files."""
        counts = count_project_files(PROJECT_ROOT)

        assert counts['source_files'] > 0
        assert counts['test_files'] > 0
        assert counts['total'] == sum(
            v for k, v in counts.items() if k != 'total'
        )

    def test_count_test_functions_on_tmp_tree(self, tmp_path):
        """Regex counts test methods inside classes and bare functions."""
        (tmp_path / "test_a.py").write_text(
            "def test_bare():\n    pass\n\n"
            "class TestX:\n"
            "    def test_one(self):\n        pass\n"
            "    def test_two(self):\n        pass\n"
        )
        (tmp_path / "test_b.py").write_text("def test_other():\n    pass\n")
        (tmp_path / "not_a_test.py").write_text("def test_ignored():\n    pass\n")

        assert count_test_functions(tmp_path) == 4


class TestReportGeneration:
    """Test the report generation entrypoint."""

    def test_generate_documentation_reports(self, tmp_path):
        """Writes three deterministic artifacts under the target dir."""
        reports_dir = tmp_path / "reports"

        result = generate_documentation_reports(PROJECT_ROOT, reports_dir)

        names = [p.name for p in result['paths']]
        assert names == ["MASTER_INDEX.json", "FILE_COUNTS.json", "SYSTEM_REPORT.md"]
        for path in result['paths']:
            assert path.exists() and path.stat().st_size > 0

        # Determinism: regenerating yields byte-identical artifacts
        first = {p.name: p.read_text() for p in result['paths']}
        generate_documentation_reports(PROJECT_ROOT, reports_dir)
        for name, content in first.items():
            assert (reports_dir / name).read_text() == content
