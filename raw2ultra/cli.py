"""Command line interface for raw2ultra."""

import click
from . import __version__


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """raw2ultra - A Python CLI tool for raw2ultra processing."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        click.echo(f"raw2ultra v{__version__}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def process(ctx, input_file, output):
    """Process a raw file to ultra format."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"Processing file: {input_file}")
        if output:
            click.echo(f"Output file: {output}")
    
    # TODO: Implement the actual processing logic here
    click.echo("Processing completed!")


@cli.command()
def info():
    """Show information about raw2ultra."""
    click.echo(f"raw2ultra version {__version__}")
    click.echo("A Python CLI tool for raw2ultra processing")


if __name__ == '__main__':
    cli() 