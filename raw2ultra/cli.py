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


def convert_to_rgba10102(rgb_data: np.ndarray) -> np.ndarray:
    """Convert RGB 16-bit data to RGBA10:10:10:2 format.

    Args:
        rgb_data: RGB data with shape (h, w, 3) and dtype uint16

    Returns:
        RGBA10:10:10:2 data with shape (h, w) and dtype uint32
    """
    click.echo("Converting to RGBA10:10:10:2 format...")

    height, width, channels = rgb_data.shape
    assert channels == 3, "Input must be RGB data with 3 channels"

    # Convert from 16-bit to 10-bit for RGB channels
    # 16-bit range: 0-65535, 10-bit range: 0-1023
    rgb_10bit = (rgb_data.astype(np.float32) / 65535.0 * 1023.0).astype(np.uint16)

    # Create alpha channel (2-bit, full opacity = 3)
    alpha_2bit = np.full((height, width), 3, dtype=np.uint16)

    # Pack into 32-bit RGBA10:10:10:2 format
    # Bit layout: [31:22] R, [21:12] G, [11:2] B, [1:0] A
    rgba_packed = (
        (rgb_10bit[:, :, 0].astype(np.uint32) << 22)  # Red: bits 31-22
        | (rgb_10bit[:, :, 1].astype(np.uint32) << 12)  # Green: bits 21-12
        | (rgb_10bit[:, :, 2].astype(np.uint32) << 2)  # Blue: bits 11-2
        | alpha_2bit.astype(np.uint32)  # Alpha: bits 1-0
    )

    click.echo(f"Converted to RGBA10:10:10:2 format: {width}x{height} pixels")

    return rgba_packed


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

    with rawpy.imread(input_file) as raw:
        # ndarray of shape (h,w,c)
        data: np.ndarray = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.P3D65,
            output_bps=16,
        )

    height, width, _ = data.shape

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

        # Resize using skimage, preserving data type
        data = resize(
            data, (new_height, new_width), preserve_range=True, anti_aliasing=True
        ).astype(data.dtype)
        height, width = new_height, new_width

    # Make dimensions even by cropping if necessary
    if width % 2 == 1:
        data = data[:, :-1, :]  # Remove last column
        width -= 1
        click.echo(f"Cropped width to make it even: {width}")

    if height % 2 == 1:
        data = data[:-1, :, :]  # Remove last row
        height -= 1
        click.echo(f"Cropped height to make it even: {height}")

    click.echo(f"Final dimensions: {width}x{height}")

    # Convert to RGBA10:10:10:2 format
    rgba_data = convert_to_rgba10102(data)

    # Determine output filename
    bin_file = input_file + ".bin"

    click.echo(f"Writing RGBA10:10:10:2 data to: {bin_file}")

    # Write as binary file
    with open(bin_file, "wb") as f:
        f.write(rgba_data.tobytes("C"))

    click.echo(f"Successfully wrote {rgba_data.size * 4} bytes to {bin_file}")

    try:
        subprocess.run(
            [
                "ultrahdr_app",
                "-m",
                "0",
                "-p",
                bin_file,
                "-w",
                str(width),
                "-h",
                str(height),
                "-q",
                "97",
                "-a",
                "5",
                "-z",
                output,
            ],
            check=True,
        )

        click.echo(f"Successfully processed {input_file} and saved to {output}")

    finally:
        click.echo(f"Cleaning up: {bin_file}")
        os.unlink(bin_file)


if __name__ == "__main__":
    cli()
