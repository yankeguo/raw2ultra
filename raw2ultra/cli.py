"""Command line interface for raw2ultra."""

import os
import click
import rawpy
import imageio.v3 as iio
import numpy as np
import subprocess
from skimage.transform import resize

from PIL import Image
from . import __version__


def convert_to_rgbahalffloat(rgb_data: np.ndarray) -> np.ndarray:
    """Convert RGB data to RGBA half-float format.

    Args:
        rgb_data: Linear RGB data with shape (h, w, 3) in 0-1 range

    Returns:
        RGBA half-float data with shape (h, w, 4) and dtype float16
    """
    click.echo("Converting to RGBA half-float format...")

    height, width, channels = rgb_data.shape
    assert channels == 3, "Input must be RGB data with 3 channels"

    # Ensure data is in 0-1 range and convert to float16
    rgb_f16 = np.clip(rgb_data, 0.0, 1.0).astype(np.float16)

    # Create alpha channel (half-float, full opacity = 1.0)
    alpha_f16 = np.ones((height, width, 1), dtype=np.float16)

    # Concatenate RGB and Alpha channels
    rgba_data = np.concatenate([rgb_f16, alpha_f16], axis=2)

    click.echo(f"Converted to RGBA half-float format: {width}x{height} pixels")

    return rgba_data


@click.command()
@click.version_option(version=__version__)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def cli(input_file, output):
    """raw2ultra - A Python CLI tool for raw2ultra processing."""

    if output:
        output = output + ".jpg"
    else:
        output = input_file + ".jpg"

    click.echo(f"raw2ultra v{__version__}")
    click.echo(f"Processing file: {input_file}")
    if output:
        click.echo(f"Output file: {output}")

    click.echo("=== Processing HDR Intent (Linear) ===")

    # Process HDR intent with linear transfer function
    with rawpy.imread(input_file) as raw:
        # ndarray of shape (h,w,c) - keep linear for rgbahalffloat
        hdr_data: np.ndarray = raw.postprocess(
            gamma=(1, 1),  # Linear gamma
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.P3D65,  # P3 color space
            output_bps=16,
        )

    # Convert to linear float data (0-1 range)
    hdr_linear_data = hdr_data.astype(np.float32) / 65535.0

    # Debug: Show HDR linear data range
    click.echo(
        f"HDR Linear data range: min={hdr_linear_data.min():.6f}, max={hdr_linear_data.max():.6f}, mean={hdr_linear_data.mean():.6f}"
    )

    height, width, _ = hdr_linear_data.shape
    click.echo(f"Original dimensions: {width}x{height}")

    # Scale down if larger than 8192x8192
    max_size = 8192
    if width > max_size or height > max_size:
        scale_factor = min(max_size / width, max_size / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        click.echo(
            f"Scaling down by factor {scale_factor:.3f} to {new_width}x{new_height}"
        )

        # Resize HDR data using skimage, preserving data type
        hdr_linear_data = resize(
            hdr_linear_data,
            (new_height, new_width),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(hdr_linear_data.dtype)

        height, width = new_height, new_width

    # Make dimensions even by cropping if necessary
    if width % 2 == 1:
        hdr_linear_data = hdr_linear_data[:, :-1, :]  # Remove last column
        width -= 1
        click.echo(f"Cropped width to make it even: {width}")

    if height % 2 == 1:
        hdr_linear_data = hdr_linear_data[:-1, :, :]  # Remove last row
        height -= 1
        click.echo(f"Cropped height to make it even: {height}")

    click.echo(f"Final dimensions: {width}x{height}")

    # Convert HDR to RGBA half-float format
    hdr_rgba_data = convert_to_rgbahalffloat(hdr_linear_data)

    # Generate file name
    hdr_bin_file = input_file + "_hdr.bin"

    click.echo(f"Writing HDR RGBA half-float data to: {hdr_bin_file}")

    # Write HDR binary file
    with open(hdr_bin_file, "wb") as f:
        f.write(hdr_rgba_data.tobytes("C"))

    click.echo(f"Successfully wrote {hdr_rgba_data.size * 2} bytes to {hdr_bin_file}")

    try:
        # Use encode scenario 0 with rgbahalffloat format
        subprocess.run(
            [
                "ultrahdr_app",
                "-m",
                "0",  # encode mode
                "-p",
                hdr_bin_file,  # HDR intent input only
                "-w",
                str(width),
                "-h",
                str(height),
                "-q",
                "98",  # Quality factor
                "-a",
                "4",  # HDR format: rgbahalffloat
                "-C",
                "1",  # HDR color gamut: p3 (DisplayP3)
                "-t",
                "0",  # HDR transfer function: linear (required for rgbahalffloat)
                "-R",
                "1",  # Full range for HDR intent
                "-z",
                output,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        click.echo(f"Successfully processed {input_file} and saved to {output}")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error processing {input_file}")
        click.echo(f"Error code: {e.returncode}")

    finally:
        click.echo(f"Cleaning up: {hdr_bin_file}")
        os.unlink(hdr_bin_file)


if __name__ == "__main__":
    cli()
