"""Enhanced logging with colors, progress bars, and better formatting."""

import sys
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import subprocess


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Additional colors
    GRAY = '\033[90m'
    LIGHT_GRAY = '\033[37m'
    YELLOW = '\033[33m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[97m'
    
    @staticmethod
    def disable():
        """Disable all colors."""
        Colors.HEADER = ''
        Colors.OKBLUE = ''
        Colors.OKCYAN = ''
        Colors.OKGREEN = ''
        Colors.WARNING = ''
        Colors.FAIL = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''
        Colors.GRAY = ''
        Colors.LIGHT_GRAY = ''
        Colors.YELLOW = ''
        Colors.MAGENTA = ''
        Colors.CYAN = ''
        Colors.WHITE = ''


class EnhancedLogger:
    """Enhanced logger with colors, progress, and formatting."""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and sys.stdout.isatty()
        if not self.use_colors:
            Colors.disable()
        
        self.start_time = time.time()
        self.step_times = {}
    
    def header(self, message: str, char: str = "=", color: str = "OKBLUE"):
        """Print a formatted header."""
        width = 80
        color_code = getattr(Colors, color, Colors.OKBLUE)
        print(f"\n{color_code}{Colors.BOLD}{char * width}")
        print(f"  {message}")
        print(f"{char * width}{Colors.ENDC}\n")
    
    def subheader(self, message: str, char: str = "─"):
        """Print a formatted subheader."""
        width = 80
        print(f"\n{Colors.CYAN}{char * width}")
        print(f"  {message}")
        print(f"{char * width}{Colors.ENDC}\n")
    
    def step(self, step: int, total: int, message: str):
        """Print a step indicator with timing."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print(f"{Colors.BOLD}[{Colors.OKCYAN}{step}/{total}{Colors.ENDC}{Colors.BOLD}] "
              f"{Colors.ENDC}{message}... "
              f"{Colors.GRAY}(elapsed: {minutes}m {seconds}s){Colors.ENDC}")
        
        self.step_times[message] = time.time()
    
    def success(self, message: str, detail: Optional[str] = None):
        """Print a success message."""
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} {message}")
        if detail:
            print(f"  {Colors.GRAY}{detail}{Colors.ENDC}")
    
    def error(self, message: str, detail: Optional[str] = None):
        """Print an error message."""
        print(f"{Colors.FAIL}✗{Colors.ENDC} {message}")
        if detail:
            print(f"  {Colors.FAIL}{detail}{Colors.ENDC}")
    
    def warning(self, message: str, detail: Optional[str] = None):
        """Print a warning message."""
        print(f"{Colors.WARNING}⚠{Colors.ENDC} {message}")
        if detail:
            print(f"  {Colors.GRAY}{detail}{Colors.ENDC}")
    
    def info(self, message: str, indent: int = 0):
        """Print an info message."""
        prefix = "  " * indent
        print(f"{prefix}{Colors.OKCYAN}▶{Colors.ENDC} {message}")
    
    def detail(self, key: str, value: Any, indent: int = 1):
        """Print a key-value detail."""
        prefix = "  " * indent
        print(f"{prefix}{Colors.LIGHT_GRAY}{key}:{Colors.ENDC} {Colors.WHITE}{value}{Colors.ENDC}")
    
    def progress_bar(
        self,
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 50
    ):
        """Display a progress bar."""
        percent = current / total
        filled = int(length * percent)
        bar = '█' * filled + '░' * (length - filled)
        
        print(f'\r{prefix} {Colors.OKCYAN}[{bar}]{Colors.ENDC} '
              f'{Colors.BOLD}{percent * 100:.1f}%{Colors.ENDC} {suffix}', 
              end='', flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def section(self, title: str):
        """Print a section divider."""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'─' * 3} {title} {'─' * (74 - len(title))}{Colors.ENDC}")
    
    def list_item(self, item: str, status: Optional[str] = None):
        """Print a list item."""
        if status == "success":
            icon = f"{Colors.OKGREEN}✓{Colors.ENDC}"
        elif status == "error":
            icon = f"{Colors.FAIL}✗{Colors.ENDC}"
        elif status == "warning":
            icon = f"{Colors.WARNING}⚠{Colors.ENDC}"
        elif status == "pending":
            icon = f"{Colors.GRAY}○{Colors.ENDC}"
        else:
            icon = "•"
        
        print(f"  {icon} {item}")
    
    def command(self, cmd: str, description: str):
        """Print a command being executed."""
        print(f"\n{Colors.BOLD}▶ {description}{Colors.ENDC}")
        print(f"  {Colors.GRAY}Command: {Colors.ENDC}{Colors.CYAN}{cmd}{Colors.ENDC}")
    
    def table_header(self, columns: list, widths: list):
        """Print a table header."""
        header = "  "
        for col, width in zip(columns, widths):
            header += f"{Colors.BOLD}{col:<{width}}{Colors.ENDC}  "
        print(header)
        print(f"  {Colors.GRAY}{'-' * sum(widths)}{Colors.ENDC}")
    
    def table_row(self, values: list, widths: list):
        """Print a table row."""
        row = "  "
        for val, width in zip(values, widths):
            row += f"{str(val):<{width}}  "
        print(row)
    
    def elapsed_time(self, label: str = "Total time"):
        """Print elapsed time."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"\n{Colors.BOLD}{label}:{Colors.ENDC} {Colors.OKGREEN}{minutes}m {seconds}s{Colors.ENDC}")
    
    def step_summary(self, step_name: str):
        """Print summary for a step."""
        if step_name in self.step_times:
            elapsed = time.time() - self.step_times[step_name]
            self.success(f"{step_name} completed", f"took {elapsed:.1f}s")
        else:
            self.success(f"{step_name} completed")
    
    def file_info(self, filepath: Path):
        """Print file information."""
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {filepath.name} "
                  f"{Colors.GRAY}({size_mb:.2f} MB){Colors.ENDC}")
        else:
            print(f"  {Colors.FAIL}✗{Colors.ENDC} {filepath.name} "
                  f"{Colors.GRAY}(not found){Colors.ENDC}")
    
    def system_info(self, info: Dict[str, Any]):
        """Print system information."""
        self.section("Environment Information")
        for key, value in info.items():
            self.detail(key, value, indent=1)
    
    def box(self, title: str, content: list, width: int = 78):
        """Print a boxed message."""
        print(f"\n{Colors.BOLD}┌{'─' * (width - 2)}┐{Colors.ENDC}")
        print(f"{Colors.BOLD}│{Colors.ENDC} {title.center(width - 4)} "
              f"{Colors.BOLD}│{Colors.ENDC}")
        print(f"{Colors.BOLD}├{'─' * (width - 2)}┤{Colors.ENDC}")
        for line in content:
            padding = width - 4 - len(line)
            print(f"{Colors.BOLD}│{Colors.ENDC} {line}{' ' * padding} "
                  f"{Colors.BOLD}│{Colors.ENDC}")
        print(f"{Colors.BOLD}└{'─' * (width - 2)}┘{Colors.ENDC}\n")
    
    def summary_box(self, items: Dict[str, str]):
        """Print a summary box with key-value pairs."""
        max_key_len = max(len(k) for k in items.keys())
        
        print(f"\n{Colors.BOLD}{'═' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'SUMMARY'.center(80)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'═' * 80}{Colors.ENDC}\n")
        
        for key, value in items.items():
            print(f"  {Colors.BOLD}{key:>{max_key_len}}:{Colors.ENDC} "
                  f"{Colors.OKGREEN}{value}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}{'═' * 80}{Colors.ENDC}\n")
    
    def metrics_table(self, metrics: Dict[str, Any], title: str = "Metrics"):
        """Print a formatted metrics table."""
        self.section(title)
        
        # Group metrics by category if they have dots
        categories = {}
        for key, value in metrics.items():
            if '.' in key:
                category, metric = key.rsplit('.', 1)
                if category not in categories:
                    categories[category] = {}
                categories[category][metric] = value
            else:
                if 'General' not in categories:
                    categories['General'] = {}
                categories['General'][key] = value
        
        for category, cat_metrics in categories.items():
            print(f"\n  {Colors.BOLD}{Colors.CYAN}{category}:{Colors.ENDC}")
            for key, value in cat_metrics.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                elif isinstance(value, int):
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)
                
                print(f"    {Colors.LIGHT_GRAY}{key}:{Colors.ENDC} "
                      f"{Colors.WHITE}{formatted_value}{Colors.ENDC}")


def create_enhanced_logger(use_colors: bool = True) -> EnhancedLogger:
    """
    Create an enhanced logger instance.
    
    Args:
        use_colors: Whether to use colored output (auto-detected for TTY)
        
    Returns:
        EnhancedLogger instance
    """
    return EnhancedLogger(use_colors=use_colors)


# Convenience function for quick messages
def log_success(message: str):
    """Quick success message."""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {message}")


def log_error(message: str):
    """Quick error message."""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {message}")


def log_warning(message: str):
    """Quick warning message."""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {message}")


def log_info(message: str):
    """Quick info message."""
    print(f"{Colors.OKCYAN}▶{Colors.ENDC} {message}")

