"""Command-line interface for EpixVanity."""

import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text

from .core.config import EpixConfig
from .core.generator import VanityGenerator
from .utils.patterns import Pattern, PatternType, PatternValidator
from .utils.logging import setup_logging, get_logger
from .gpu import CUDA_AVAILABLE, OPENCL_AVAILABLE

if CUDA_AVAILABLE:
    from .gpu.cuda_generator import CudaVanityGenerator

if OPENCL_AVAILABLE:
    from .gpu.opencl_generator import OpenCLVanityGenerator


console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, log_file: Optional[str], config: Optional[str]):
    """EpixVanity - High-performance vanity address generator for Epix blockchain."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, log_file=log_file)
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            config_data = json.load(f)
        epix_config = EpixConfig.from_dict(config_data)
    else:
        epix_config = EpixConfig.default_testnet()
    
    # Store in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = epix_config
    ctx.obj['logger'] = get_logger()


@cli.command()
@click.argument('pattern')
@click.option('--type', 'pattern_type', 
              type=click.Choice(['prefix', 'suffix', 'contains', 'regex']),
              default='prefix',
              help='Pattern type')
@click.option('--case-sensitive', is_flag=True, help='Case sensitive matching')
@click.option('--max-attempts', type=int, help='Maximum number of attempts')
@click.option('--timeout', type=float, help='Timeout in seconds')
@click.option('--threads', type=int, help='Number of CPU threads')
@click.option('--gpu', type=click.Choice(['cuda', 'opencl']), help='Use GPU acceleration')
@click.option('--gpu-device', type=int, default=0, help='GPU device ID')
@click.option('--batch-size', type=int, default=1024*1024, help='GPU batch size')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--format', 'output_format', 
              type=click.Choice(['json', 'text', 'csv']),
              default='text',
              help='Output format')
@click.pass_context
def generate(
    ctx: click.Context,
    pattern: str,
    pattern_type: str,
    case_sensitive: bool,
    max_attempts: Optional[int],
    timeout: Optional[float],
    threads: Optional[int],
    gpu: Optional[str],
    gpu_device: int,
    batch_size: int,
    output: Optional[str],
    output_format: str
):
    """Generate a vanity address matching the specified pattern."""
    
    config = ctx.obj['config']
    logger = ctx.obj['logger']
    
    try:
        # Create pattern
        pattern_validator = PatternValidator(config.get_address_prefix())
        pattern_obj = pattern_validator.create_pattern(
            pattern, 
            PatternType(pattern_type),
            case_sensitive
        )
        
        # Display pattern information
        pattern_info = pattern_validator.get_pattern_info(pattern_obj)
        
        console.print(Panel.fit(
            f"[bold blue]Pattern:[/bold blue] {pattern}\n"
            f"[bold blue]Type:[/bold blue] {pattern_type}\n"
            f"[bold blue]Case Sensitive:[/bold blue] {case_sensitive}\n"
            f"[bold blue]Estimated Difficulty:[/bold blue] {pattern_info['difficulty_description']}\n"
            f"[bold blue]Expected Attempts:[/bold blue] {pattern_info['estimated_attempts']:,}",
            title="Vanity Generation Settings"
        ))
        
        # Choose generator
        if gpu == 'cuda':
            if not CUDA_AVAILABLE:
                console.print("[red]CUDA is not available. Install PyCUDA to use CUDA acceleration.[/red]")
                sys.exit(1)
            
            generator = CudaVanityGenerator(
                config=config,
                device_id=gpu_device,
                batch_size=batch_size
            )
            console.print(f"[green]Using CUDA acceleration on device {gpu_device}[/green]")
            
        elif gpu == 'opencl':
            if not OPENCL_AVAILABLE:
                console.print("[red]OpenCL is not available. Install PyOpenCL to use OpenCL acceleration.[/red]")
                sys.exit(1)
            
            generator = OpenCLVanityGenerator(
                config=config,
                device_id=gpu_device,
                batch_size=batch_size
            )
            console.print(f"[green]Using OpenCL acceleration on device {gpu_device}[/green]")
            
        else:
            generator = VanityGenerator(
                config=config,
                num_threads=threads
            )
            console.print(f"[green]Using CPU generation with {generator.num_threads} threads[/green]")
        
        # Start generation with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            
            task = progress.add_task("Generating vanity address...", total=None)
            
            def progress_callback(stats: Dict[str, Any]) -> None:
                progress.update(
                    task,
                    description=f"Rate: {stats.get('Current Rate', '0')} | "
                               f"Attempts: {stats.get('Total Attempts', '0')}"
                )
            
            # Set progress callback if using CPU generator
            if hasattr(generator, 'progress_callback'):
                generator.progress_callback = progress_callback
            
            # Generate vanity address
            result = generator.generate_vanity_address(
                pattern_obj,
                max_attempts=max_attempts,
                timeout=timeout
            )
        
        # Display results
        if result.success:
            console.print("\n[bold green]🎉 SUCCESS![/bold green]")
            
            # Create results table
            table = Table(title="Vanity Address Generated")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Address", result.keypair.address)
            table.add_row("Ethereum Address", result.keypair.eth_address)
            table.add_row("Private Key", generator.crypto.private_key_to_hex(result.keypair.private_key))
            table.add_row("Attempts", f"{result.attempts:,}")
            table.add_row("Time Elapsed", f"{result.elapsed_time:.2f}s")
            
            if result.attempts > 0:
                rate = result.attempts / result.elapsed_time
                table.add_row("Average Rate", f"{rate:.0f} attempts/s")
            
            console.print(table)
            
            # Save to file if requested
            if output:
                save_result(result, output, output_format, pattern)
                console.print(f"\n[green]Results saved to {output}[/green]")
        
        else:
            console.print(f"\n[red]❌ FAILED: {result.error}[/red]")
            if result.attempts > 0:
                console.print(f"Attempted {result.attempts:,} addresses in {result.elapsed_time:.2f}s")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx: click.Context):
    """Display system and device information."""
    
    config = ctx.obj['config']
    
    # System info
    console.print(Panel.fit(
        f"[bold blue]Chain ID:[/bold blue] {config.chain_id}\n"
        f"[bold blue]Chain Name:[/bold blue] {config.chain_name}\n"
        f"[bold blue]Address Prefix:[/bold blue] {config.get_address_prefix()}\n"
        f"[bold blue]RPC:[/bold blue] {config.rpc}\n"
        f"[bold blue]REST API:[/bold blue] {config.rest}",
        title="Epix Configuration"
    ))
    
    # CPU info
    import os
    import psutil
    
    cpu_table = Table(title="CPU Information")
    cpu_table.add_column("Property", style="cyan")
    cpu_table.add_column("Value", style="white")
    
    cpu_table.add_row("CPU Count", str(os.cpu_count()))
    cpu_table.add_row("CPU Usage", f"{psutil.cpu_percent()}%")
    cpu_table.add_row("Memory Usage", f"{psutil.virtual_memory().percent}%")
    cpu_table.add_row("Available Memory", f"{psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    console.print(cpu_table)
    
    # GPU info
    if CUDA_AVAILABLE:
        try:
            from .gpu.cuda_generator import CudaVanityGenerator
            cuda_gen = CudaVanityGenerator()
            cuda_info = cuda_gen.get_device_info()
            
            cuda_table = Table(title="CUDA Information")
            cuda_table.add_column("Property", style="cyan")
            cuda_table.add_column("Value", style="green")
            
            for key, value in cuda_info.items():
                if key != "error":
                    cuda_table.add_row(key.replace("_", " ").title(), str(value))
            
            console.print(cuda_table)
            
        except Exception as e:
            console.print(f"[yellow]CUDA Error: {e}[/yellow]")
    else:
        console.print("[yellow]CUDA not available[/yellow]")
    
    if OPENCL_AVAILABLE:
        try:
            from .gpu.opencl_generator import OpenCLVanityGenerator
            devices = OpenCLVanityGenerator.list_devices()
            
            if devices and "error" not in devices[0]:
                opencl_table = Table(title="OpenCL Devices")
                opencl_table.add_column("Platform", style="cyan")
                opencl_table.add_column("Device", style="cyan")
                opencl_table.add_column("Name", style="white")
                opencl_table.add_column("Type", style="green")
                opencl_table.add_column("Compute Units", style="yellow")
                
                for device in devices:
                    opencl_table.add_row(
                        str(device.get("platform_id", "N/A")),
                        str(device.get("device_id", "N/A")),
                        device.get("device_name", "Unknown"),
                        device.get("device_type", "Unknown"),
                        str(device.get("max_compute_units", "N/A"))
                    )
                
                console.print(opencl_table)
            else:
                console.print("[yellow]No OpenCL devices found[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]OpenCL Error: {e}[/yellow]")
    else:
        console.print("[yellow]OpenCL not available[/yellow]")


@cli.command()
@click.argument('word')
@click.pass_context
def suggest(ctx: click.Context, word: str):
    """Suggest valid patterns based on a word."""
    
    config = ctx.obj['config']
    pattern_validator = PatternValidator(config.get_address_prefix())
    
    suggestions = pattern_validator.suggest_patterns(word)
    
    if suggestions:
        console.print(f"\n[bold blue]Suggestions for '{word}':[/bold blue]")
        
        for suggestion in suggestions:
            pattern_obj = pattern_validator.create_pattern(suggestion, PatternType.PREFIX)
            pattern_info = pattern_validator.get_pattern_info(pattern_obj)
            
            console.print(f"  [green]{suggestion}[/green] - {pattern_info['difficulty_description']} "
                         f"({pattern_info['estimated_attempts']:,} attempts)")
    else:
        console.print(f"[yellow]No valid patterns found for '{word}'[/yellow]")


def save_result(result, output_path: str, format_type: str, pattern: str) -> None:
    """Save generation result to file."""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == 'json':
        data = {
            "success": result.success,
            "pattern": pattern,
            "address": result.keypair.address if result.keypair else None,
            "eth_address": result.keypair.eth_address if result.keypair else None,
            "private_key": result.keypair.private_key.hex() if result.keypair else None,
            "attempts": result.attempts,
            "elapsed_time": result.elapsed_time,
            "timestamp": time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format_type == 'csv':
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Pattern', 'Address', 'Ethereum Address', 'Private Key', 'Attempts', 'Time'])
            
            if result.success:
                writer.writerow([
                    pattern,
                    result.keypair.address,
                    result.keypair.eth_address,
                    result.keypair.private_key.hex(),
                    result.attempts,
                    result.elapsed_time
                ])
    
    else:  # text format
        with open(output_file, 'w') as f:
            f.write(f"EpixVanity Generation Result\n")
            f.write(f"===========================\n\n")
            f.write(f"Pattern: {pattern}\n")
            f.write(f"Success: {result.success}\n")
            
            if result.success:
                f.write(f"Address: {result.keypair.address}\n")
                f.write(f"Ethereum Address: {result.keypair.eth_address}\n")
                f.write(f"Private Key: {result.keypair.private_key.hex()}\n")
            
            f.write(f"Attempts: {result.attempts:,}\n")
            f.write(f"Time Elapsed: {result.elapsed_time:.2f}s\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
