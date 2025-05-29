"""Command line interface for raw2ultra."""

import click
from . import __version__


@click.command()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def cli(verbose, input_file, output):
    """raw2ultra - A Python CLI tool for raw2ultra processing."""
    
    if verbose:
        click.echo(f"raw2ultra v{__version__}")
        click.echo(f"Processing file: {input_file}")
        if output:
            click.echo(f"Output file: {output}")
    
    # TODO: Implement the actual processing logic here
    click.echo("Processing completed!")


if __name__ == '__main__':
    cli() 