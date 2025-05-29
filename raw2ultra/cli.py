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


def apply_hlg_curve(linear_data: np.ndarray) -> np.ndarray:
    """Apply HLG (Hybrid Log-Gamma) curve to linear data

    Args:
        linear_data: Linear RGB data normalized to 0-1 range

    Returns:
        HLG-encoded data in 0-1 range
    """
    # HLG parameters (ITU-R BT.2100)
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    # Ensure data is in 0-1 range
    normalized = np.clip(linear_data, 0.0, 1.0)

    # Calculate the threshold (1/12)
    threshold = 1.0 / 12.0

    # For the log portion, ensure the argument is always positive
    log_arg = 12.0 * normalized - b
    # Clamp to a small positive value to avoid log(0) or log(negative)
    log_arg = np.maximum(log_arg, 1e-10)

    # Apply HLG curve according to ITU-R BT.2100
    hlg_data = np.where(
        normalized <= threshold,
        np.sqrt(3.0 * normalized),  # Linear portion: sqrt(3*E) for E <= 1/12
        a * np.log(log_arg) + c,  # Log portion: a*ln(12*E - b) + c for E > 1/12
    )

    # Ensure output is in valid range
    hlg_data = np.clip(hlg_data, 0.0, 1.0)

    return hlg_data


def convert_to_rgba10102(rgb_data: np.ndarray) -> np.ndarray:
    """Convert RGB data to RGBA10:10:10:2 format.

    Args:
        rgb_data: RGB data with shape (h, w, 3) in 0-1 range (HLG-encoded)

    Returns:
        RGBA10:10:10:2 data with shape (h, w) and dtype uint32
    """
    click.echo("Converting to RGBA10:10:10:2 format...")

    height, width, channels = rgb_data.shape
    assert channels == 3, "Input must be RGB data with 3 channels"

    # Convert from 0-1 range to 10-bit for RGB channels
    # Input range: 0-1, 10-bit range: 0-1023
    rgb_10bit = np.clip(rgb_data * 1023.0, 0, 1023).astype(np.uint16)

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


def convert_to_rgba8888(rgb_data: np.ndarray) -> np.ndarray:
    """Convert RGB data to RGBA8888 format.

    Args:
        rgb_data: RGB data with shape (h, w, 3) in 0-1 range (sRGB-encoded)

    Returns:
        RGBA8888 data with shape (h, w, 4) and dtype uint8
    """
    click.echo("Converting to RGBA8888 format...")

    height, width, channels = rgb_data.shape
    assert channels == 3, "Input must be RGB data with 3 channels"

    # Convert from 0-1 range to 8-bit for RGB channels
    # Input range: 0-1, 8-bit range: 0-255
    rgb_8bit = np.clip(rgb_data * 255.0, 0, 255).astype(np.uint8)

    # Create alpha channel (8-bit, full opacity = 255)
    alpha_8bit = np.full((height, width, 1), 255, dtype=np.uint8)

    # Concatenate RGB and Alpha channels
    rgba_data = np.concatenate([rgb_8bit, alpha_8bit], axis=2)

    click.echo(f"Converted to RGBA8888 format: {width}x{height} pixels")

    return rgba_data


def apply_srgb_curve(linear_data: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma curve to linear data

    Args:
        linear_data: Linear RGB data normalized to 0-1 range

    Returns:
        sRGB-encoded data in 0-1 range
    """
    # Ensure data is in 0-1 range
    normalized = np.clip(linear_data, 0.0, 1.0)

    # Apply sRGB curve
    # For values <= 0.0031308: output = 12.92 * input
    # For values > 0.0031308: output = 1.055 * (input^(1/2.4)) - 0.055
    srgb_data = np.where(
        normalized <= 0.0031308,
        12.92 * normalized,
        1.055 * np.power(normalized, 1.0 / 2.4) - 0.055,
    )

    # Ensure output is in valid range
    srgb_data = np.clip(srgb_data, 0.0, 1.0)

    return srgb_data


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

    click.echo("=== Processing HDR Intent ===")

    # Process HDR intent (Rec2020, HLG)
    with rawpy.imread(input_file) as raw:
        # ndarray of shape (h,w,c)
        hdr_data: np.ndarray = raw.postprocess(
            gamma=(1, 1),
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.Rec2020,
            output_bps=16,
        )

    hdr_linear_data = hdr_data.astype(np.float32) / 65535.0

    # Debug: Show HDR linear data range
    click.echo(
        f"HDR Linear data range: min={hdr_linear_data.min():.6f}, max={hdr_linear_data.max():.6f}, mean={hdr_linear_data.mean():.6f}"
    )

    # Apply HLG curve to linear data
    hlg_data = apply_hlg_curve(hdr_linear_data)

    # Debug: Show HLG data range
    click.echo(
        f"HLG data range: min={hlg_data.min():.6f}, max={hlg_data.max():.6f}, mean={hlg_data.mean():.6f}"
    )

    click.echo("=== Processing SDR Intent ===")

    # Process SDR intent (BT709, sRGB)
    with rawpy.imread(input_file) as raw:
        # ndarray of shape (h,w,c)
        sdr_data: np.ndarray = raw.postprocess(
            gamma=(2.222, 4.5),  # sRGB gamma
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
        )

    # Convert to linear first, then apply sRGB curve for better control
    sdr_linear_data = sdr_data.astype(np.float32) / 65535.0

    # Debug: Show SDR linear data range
    click.echo(
        f"SDR Linear data range: min={sdr_linear_data.min():.6f}, max={sdr_linear_data.max():.6f}, mean={sdr_linear_data.mean():.6f}"
    )

    # Apply sRGB curve
    srgb_data = apply_srgb_curve(sdr_linear_data)

    # Debug: Show sRGB data range
    click.echo(
        f"sRGB data range: min={srgb_data.min():.6f}, max={srgb_data.max():.6f}, mean={srgb_data.mean():.6f}"
    )

    height, width, _ = hlg_data.shape

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

        # Resize both HDR and SDR data using skimage, preserving data type
        hlg_data = resize(
            hlg_data, (new_height, new_width), preserve_range=True, anti_aliasing=True
        ).astype(hlg_data.dtype)

        srgb_data = resize(
            srgb_data, (new_height, new_width), preserve_range=True, anti_aliasing=True
        ).astype(srgb_data.dtype)

        height, width = new_height, new_width

    # Make dimensions even by cropping if necessary
    if width % 2 == 1:
        hlg_data = hlg_data[:, :-1, :]  # Remove last column
        srgb_data = srgb_data[:, :-1, :]  # Remove last column
        width -= 1
        click.echo(f"Cropped width to make it even: {width}")

    if height % 2 == 1:
        hlg_data = hlg_data[:-1, :, :]  # Remove last row
        srgb_data = srgb_data[:-1, :, :]  # Remove last row
        height -= 1
        click.echo(f"Cropped height to make it even: {height}")

    click.echo(f"Final dimensions: {width}x{height}")

    # Convert HDR to RGBA10:10:10:2 format
    hdr_rgba_data = convert_to_rgba10102(hlg_data)

    # Convert SDR to RGBA8888 format
    sdr_rgba_data = convert_to_rgba8888(srgb_data)

    # Generate file names
    hdr_bin_file = input_file + "_hdr.bin"
    sdr_bin_file = input_file + "_sdr.bin"

    click.echo(f"Writing HDR RGBA10:10:10:2 data to: {hdr_bin_file}")

    # Write HDR binary file
    with open(hdr_bin_file, "wb") as f:
        f.write(hdr_rgba_data.tobytes("C"))

    click.echo(f"Successfully wrote {hdr_rgba_data.size * 4} bytes to {hdr_bin_file}")

    click.echo(f"Writing SDR RGBA8888 data to: {sdr_bin_file}")

    # Write SDR binary file
    with open(sdr_bin_file, "wb") as f:
        f.write(sdr_rgba_data.tobytes("C"))

    click.echo(f"Successfully wrote {sdr_rgba_data.size * 4} bytes to {sdr_bin_file}")

    try:
        # Use encode scenario 1 parameters
        subprocess.run(
            [
                "ultrahdr_app",
                "-m",
                "0",  # encode mode
                "-p",
                hdr_bin_file,  # HDR intent input
                "-y",
                sdr_bin_file,  # SDR intent input
                "-w",
                str(width),
                "-h",
                str(height),
                "-q",
                "98",  # Quality for SDR intent
                "-Q",
                "98",  # Quality for gainmap
                "-a",
                "5",  # HDR format: rgba1010102
                "-b",
                "3",  # SDR format: rgba8888
                "-C",
                "2",  # HDR color gamut: bt2100 (matches Rec.2020)
                "-c",
                "0",  # SDR color gamut: bt709
                "-t",
                "1",  # HDR transfer function: HLG
                "-R",
                "1",  # Full range for HDR intent
                "-L",
                "1000",  # Target display brightness for HLG (1000 nits)
                "-G",
                "1.0",  # Default gamma for gainmap
                "-M",
                "1",  # Enable multi-channel gainmap
                "-D",
                "1",  # Best quality preset
                "-s",
                "1",  # Gainmap downsample factor (1 = no downsampling)
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
        click.echo(f"Cleaning up: {sdr_bin_file}")
        os.unlink(sdr_bin_file)


if __name__ == "__main__":
    cli()
