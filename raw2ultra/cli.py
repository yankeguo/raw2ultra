"""Command line interface for raw2ultra."""

import os
import click
import rawpy
import imageio.v3 as iio
import numpy as np
import struct
import io
from skimage.transform import resize

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
from . import __version__


def tone_map_to_sdr(hdr_linear_data: np.ndarray) -> np.ndarray:
    """Apply tone mapping to convert HDR linear data to SDR sRGB.

    Args:
        hdr_linear_data: Linear HDR data with shape (h, w, 3) in 0-1+ range

    Returns:
        SDR sRGB data with shape (h, w, 3) in 0-1 range
    """
    click.echo("Tone mapping HDR to SDR...")

    # Simple Reinhard tone mapping
    # Formula: L_out = L_in / (1 + L_in)
    # This compresses the dynamic range while preserving local contrast

    # Calculate luminance using Rec. 709 weights
    luminance = (
        0.2126 * hdr_linear_data[:, :, 0]
        + 0.7152 * hdr_linear_data[:, :, 1]
        + 0.0722 * hdr_linear_data[:, :, 2]
    )

    # Apply Reinhard tone mapping to luminance
    tone_mapped_luminance = luminance / (1.0 + luminance)

    # Avoid division by zero
    luminance_safe = np.maximum(luminance, 1e-8)

    # Scale RGB channels proportionally
    scale_factor = tone_mapped_luminance / luminance_safe
    scale_factor = np.expand_dims(scale_factor, axis=2)  # Add channel dimension

    tone_mapped_linear = hdr_linear_data * scale_factor

    # Apply sRGB gamma correction (linear to sRGB)
    sdr_srgb = np.where(
        tone_mapped_linear <= 0.0031308,
        12.92 * tone_mapped_linear,
        1.055 * np.power(tone_mapped_linear, 1.0 / 2.4) - 0.055,
    )

    # Ensure values are in valid range
    sdr_srgb = np.clip(sdr_srgb, 0.0, 1.0)

    click.echo(
        f"SDR range: min={sdr_srgb.min():.6f}, max={sdr_srgb.max():.6f}, mean={sdr_srgb.mean():.6f}"
    )

    return sdr_srgb


def calculate_gain_map(
    hdr_linear_data: np.ndarray, sdr_srgb_data: np.ndarray
) -> np.ndarray:
    """Calculate gain map from HDR and SDR data.

    Args:
        hdr_linear_data: Linear HDR data with shape (h, w, 3)
        sdr_srgb_data: sRGB SDR data with shape (h, w, 3)

    Returns:
        Gain map data with shape (h, w, 3) representing HDR/SDR ratio
    """
    click.echo("Calculating gain map...")

    # Convert SDR sRGB back to linear for gain calculation
    sdr_linear = np.where(
        sdr_srgb_data <= 0.04045,
        sdr_srgb_data / 12.92,
        np.power((sdr_srgb_data + 0.055) / 1.055, 2.4),
    )

    # Calculate gain = HDR / SDR (with safety for division by zero)
    sdr_linear_safe = np.maximum(sdr_linear, 1e-8)
    gain_linear = hdr_linear_data / sdr_linear_safe

    # Clamp gain to reasonable range (e.g., 1/16 to 16x)
    gain_linear = np.clip(gain_linear, 1.0 / 16.0, 16.0)

    # Convert gain to logarithmic space for better representation
    # Using log2 so that 1x gain = 0, 2x gain = 1, 4x gain = 2, etc.
    gain_log = np.log2(gain_linear)

    # Normalize to 0-1 range for JPEG encoding
    # Assuming gain range of 1/16x to 16x gives log2 range of -4 to 4
    gain_normalized = (gain_log + 4.0) / 8.0
    gain_normalized = np.clip(gain_normalized, 0.0, 1.0)

    click.echo(
        f"Gain map range: min={gain_normalized.min():.6f}, max={gain_normalized.max():.6f}, mean={gain_normalized.mean():.6f}"
    )

    return gain_normalized


def create_ultra_hdr_jpeg(
    sdr_image_data: bytes,
    gain_map_data: bytes,
    width: int,
    height: int,
    exif_data: bytes = b"",
) -> bytes:
    """Create UltraHDR JPEG with embedded gain map using MPF (Multi-Picture Format).

    Args:
        sdr_image_data: JPEG bytes of the SDR image
        gain_map_data: JPEG bytes of the gain map
        width: Image width
        height: Image height
        exif_data: Optional EXIF data to include

    Returns:
        UltraHDR JPEG bytes
    """
    click.echo("Creating UltraHDR JPEG with embedded gain map...")

    # Google UltraHDR XMP metadata template (Container format)
    xmp_template = """<x:xmpmeta
  xmlns:x="adobe:ns:meta/"
  x:xmptk="Adobe XMP Core 5.1.2">
  <rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:Container="http://ns.google.com/photos/1.0/container/"
      xmlns:Item="http://ns.google.com/photos/1.0/container/item/"
      xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
      hdrgm:Version="1.0">
      <Container:Directory>
        <rdf:Seq>
          <rdf:li
            rdf:parseType="Resource">
            <Container:Item
              Item:Semantic="Primary"
              Item:Mime="image/jpeg"/>
          </rdf:li>
          <rdf:li
            rdf:parseType="Resource">
            <Container:Item
              Item:Semantic="GainMap"
              Item:Mime="image/jpeg"
              Item:Length="{gain_map_size}"/>
          </rdf:li>
        </rdf:Seq>
      </Container:Directory>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""

    # Build the final JPEG
    result = bytearray()

    # Start with JPEG SOI marker
    result.extend(b"\xff\xd8")

    # Parse SDR JPEG to extract segments
    sdr_segments = []
    pos = 2  # Skip SOI
    sdr_data = sdr_image_data

    # Parse JPEG segments
    while pos < len(sdr_data) - 1:
        if sdr_data[pos] != 0xFF:
            break
        marker = sdr_data[pos + 1]
        pos += 2

        if marker == 0xD8:  # SOI
            continue
        elif marker == 0xD9:  # EOI
            break
        elif marker in [
            0xD0,
            0xD1,
            0xD2,
            0xD3,
            0xD4,
            0xD5,
            0xD6,
            0xD7,
        ]:  # RST markers (no length)
            sdr_segments.append((marker, b""))
            continue
        else:
            # Read length
            if pos + 2 > len(sdr_data):
                break
            length = struct.unpack(">H", sdr_data[pos : pos + 2])[0]
            if pos + length > len(sdr_data):
                break
            segment_data = sdr_data[pos : pos + length]
            sdr_segments.append((marker, segment_data))
            pos += length

    # Add essential JPEG segments first - skip JFIF since good.jpg doesn't have it
    # for marker, data in sdr_segments:
    #     if marker in [0xE0]:  # JFIF first
    #         result.extend(b"\xff" + bytes([marker]))
    #         result.extend(data)
    #         break

    # Calculate gain map size first
    gain_map_size = len(gain_map_data)

    # Add EXIF data if provided
    if exif_data:
        exif_segment = struct.pack(">H", len(exif_data) + 2) + exif_data
        result.extend(b"\xff\xe1")  # APP1 marker for EXIF
        result.extend(exif_segment)
        click.echo("Added EXIF data to output JPEG")

    # Add XMP metadata with gain map info
    xmp_data = xmp_template.format(gain_map_size=gain_map_size).encode("utf-8")
    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    xmp_segment = (
        struct.pack(">H", len(xmp_header) + len(xmp_data) + 2) + xmp_header + xmp_data
    )
    result.extend(b"\xff\xe1")  # APP1 marker for XMP
    result.extend(xmp_segment)

    # Create MPF header with proper TIFF IFD structure
    # MPF uses APP2 marker (0xFFE2)
    mpf_signature = b"MPF\x00"

    # Calculate the offset for the gain map
    # The gain map should start right after the main image ends (at the EOI marker)
    # We need to calculate the size of everything up to but not including the gain map
    main_image_end_offset = len(result) + 2  # +2 for the EOI marker we'll add
    gain_map_offset = main_image_end_offset

    # Create proper TIFF IFD structure for MPF
    # TIFF header: byte order + magic number + IFD offset
    tiff_header = b"MM\x00\x2a\x00\x00\x00\x08"  # Big-endian TIFF, IFD at offset 8

    # Create MPF Image entries (16 bytes each)
    # First image entry (main image) - should match good.jpg format
    # Calculate the main image size (from SOI to EOI, excluding gain map)
    main_image_size = len(sdr_image_data)

    image_entry_1 = struct.pack(
        ">I", 0x030000
    )  # Image type: 0x030000 (matches good.jpg format)
    image_entry_1 += struct.pack(">I", main_image_size)  # Size of main image (not 0)
    image_entry_1 += struct.pack(">I", 0)  # Offset (0 for first image)
    image_entry_1 += struct.pack(">H", 0)  # Dependent images count (0, not 1)
    image_entry_1 += struct.pack(">H", 0)  # Reserved

    # Second image entry (gain map) - offset will be updated later
    image_entry_2 = struct.pack(">I", 0x000000)  # Image type: undefined
    image_entry_2 += struct.pack(">I", gain_map_size)  # Size
    image_entry_2 += struct.pack(">I", 0)  # Offset (placeholder, will be calculated)
    image_entry_2 += struct.pack(">H", 0)  # Dependent images count
    image_entry_2 += struct.pack(">H", 0)  # Reserved

    # Create MP Entry array
    mp_entry_data = image_entry_1 + image_entry_2

    # Calculate the offset where MP Entry data will be stored
    # IFD has 3 entries (12 bytes each) + 2 bytes for count + 4 bytes for next IFD offset
    mp_entry_offset = 8 + 2 + (3 * 12) + 4  # Relative to start of TIFF data

    # Create IFD entries (12 bytes each)
    ifd_entries = bytearray()

    # MPF Version tag (0xB000)
    ifd_entries.extend(struct.pack(">H", 0xB000))  # Tag
    ifd_entries.extend(struct.pack(">H", 7))  # Type: UNDEFINED
    ifd_entries.extend(struct.pack(">I", 4))  # Count
    ifd_entries.extend(b"0100")  # Value: Version 1.0

    # Number of Images tag (0xB001)
    ifd_entries.extend(struct.pack(">H", 0xB001))  # Tag
    ifd_entries.extend(struct.pack(">H", 4))  # Type: LONG
    ifd_entries.extend(struct.pack(">I", 1))  # Count
    ifd_entries.extend(struct.pack(">I", 2))  # Value: 2 images

    # MP Entry tag (0xB002) - points to the MP Entry data
    ifd_entries.extend(struct.pack(">H", 0xB002))  # Tag
    ifd_entries.extend(struct.pack(">H", 7))  # Type: UNDEFINED
    ifd_entries.extend(struct.pack(">I", 32))  # Count: 32 bytes (2 entries × 16 bytes)
    ifd_entries.extend(struct.pack(">I", mp_entry_offset))  # Offset to MP Entry data

    # Build complete TIFF IFD
    tiff_ifd = bytearray()
    tiff_ifd.extend(struct.pack(">H", 3))  # Number of IFD entries
    tiff_ifd.extend(ifd_entries)  # IFD entries
    tiff_ifd.extend(struct.pack(">I", 0))  # Offset to next IFD (0 = no next IFD)
    tiff_ifd.extend(mp_entry_data)  # MP Entry data

    # Complete MPF header
    mpf_header = mpf_signature + tiff_header + tiff_ifd
    mpf_segment = struct.pack(">H", len(mpf_header) + 2) + mpf_header

    result.extend(b"\xff\xe2")  # APP2 marker for MPF
    result.extend(mpf_segment)

    # Add other essential JPEG segments
    for marker, data in sdr_segments:
        if marker in [0xE1, 0xDB, 0xC0, 0xC4]:  # EXIF, DQT, SOF, DHT
            result.extend(b"\xff" + bytes([marker]))
            result.extend(data)

    # Find and add scan data
    pos = 2
    while pos < len(sdr_data) - 1:
        if sdr_data[pos : pos + 2] == b"\xff\xda":  # SOS (Start of Scan)
            # Find the end of the scan data
            scan_start = pos
            pos += 2
            if pos + 2 <= len(sdr_data):
                length = struct.unpack(">H", sdr_data[pos : pos + 2])[0]
                pos += length

                # Find the end of the actual scan data (before next marker or EOI)
                scan_end = len(sdr_data)
                for i in range(pos, len(sdr_data) - 1):
                    if sdr_data[i] == 0xFF and sdr_data[i + 1] not in [0x00]:
                        if sdr_data[i + 1] == 0xD9:  # EOI
                            scan_end = i
                        break

                # Add the scan data
                result.extend(sdr_data[scan_start:scan_end])
                break
            break
        pos += 1

    # Calculate the offset for the gain map
    # The gain map should start right after the main image ends (at the EOI marker)
    # We need to calculate the size of everything up to but not including the gain map
    main_image_end_offset = len(result) + 2  # +2 for the EOI marker we'll add
    gain_map_offset = main_image_end_offset

    # Update the MPF directory with the correct offset
    # Find the MPF segment and update the offset
    mpf_start = result.find(b"MPF\x00")
    if mpf_start != -1:
        # Find the offset field for the second image (gain map) in the MP Entry data
        # Structure: MPF signature (4) + TIFF header (8) + IFD count (2) + IFD entries (36) + next IFD offset (4) + first entry (16) + second entry offset field starts at byte 8 of second entry
        offset_pos = (
            mpf_start + 4 + 8 + 2 + 36 + 4 + 16 + 8
        )  # Position of offset field in second MP Entry
        if offset_pos + 4 <= len(result):
            # Update the offset to point to where the gain map actually starts
            struct.pack_into(">I", result, offset_pos, gain_map_offset)

    # Add EOI marker for main image
    result.extend(b"\xff\xd9")

    # Add gain map as second image
    result.extend(gain_map_data)

    click.echo(f"Created UltraHDR JPEG: {len(result)} bytes")
    return bytes(result)


def verify_ultra_hdr_jpeg(file_path: str) -> dict:
    """Verify and extract information from an UltraHDR JPEG file.

    Args:
        file_path: Path to the UltraHDR JPEG file

    Returns:
        Dictionary with information about the file
    """
    info = {
        "is_ultra_hdr": False,
        "has_xmp_metadata": False,
        "has_mpf_structure": False,
        "num_images": 0,
        "gain_map_size": 0,
        "file_size": 0,
    }

    try:
        with open(file_path, "rb") as f:
            data = f.read()
            info["file_size"] = len(data)

        pos = 0
        if data[pos : pos + 2] != b"\xff\xd8":
            return info

        pos = 2
        while pos < len(data) - 1:
            if data[pos] != 0xFF:
                break

            marker = data[pos + 1]
            pos += 2

            if marker == 0xD9:  # EOI
                break
            elif marker in [
                0xD0,
                0xD1,
                0xD2,
                0xD3,
                0xD4,
                0xD5,
                0xD6,
                0xD7,
            ]:  # RST markers
                continue

            if pos + 2 > len(data):
                break

            length = struct.unpack(">H", data[pos : pos + 2])[0]
            if pos + length > len(data):
                break

            segment_data = data[pos : pos + length]

            # Check for XMP metadata
            if marker == 0xE1 and b"http://ns.adobe.com/xap/1.0/" in segment_data:
                info["has_xmp_metadata"] = True
                if b"hdrgm:" in segment_data:
                    info["is_ultra_hdr"] = True

            # Check for MPF structure (APP2 marker with MPF signature)
            elif (
                marker == 0xE2
                and len(segment_data) >= 6
                and segment_data[2:6] == b"MPF\x00"
            ):
                info["has_mpf_structure"] = True
                # Parse MPF directory with proper TIFF IFD structure
                mpf_data = segment_data[6:]  # Skip length and MPF signature
                if (
                    len(mpf_data) >= 12
                ):  # Minimum: TIFF header (8) + IFD count (2) + at least part of an entry
                    # Skip TIFF header (8 bytes) and read IFD count
                    if len(mpf_data) >= 10:
                        ifd_count = struct.unpack(">H", mpf_data[8:10])[0]

                        # Look for the NumberOfImages tag (0xB001)
                        ifd_start = 10
                        for i in range(ifd_count):
                            entry_offset = ifd_start + (i * 12)
                            if entry_offset + 12 <= len(mpf_data):
                                tag = struct.unpack(
                                    ">H", mpf_data[entry_offset : entry_offset + 2]
                                )[0]
                                if tag == 0xB001:  # NumberOfImages tag
                                    tag_type = struct.unpack(
                                        ">H",
                                        mpf_data[entry_offset + 2 : entry_offset + 4],
                                    )[0]
                                    tag_count = struct.unpack(
                                        ">I",
                                        mpf_data[entry_offset + 4 : entry_offset + 8],
                                    )[0]
                                    if (
                                        tag_type == 4 and tag_count == 1
                                    ):  # LONG type, single value
                                        num_images = struct.unpack(
                                            ">I",
                                            mpf_data[
                                                entry_offset + 8 : entry_offset + 12
                                            ],
                                        )[0]
                                        info["num_images"] = num_images
                                        break

                        # Look for the MP Entry tag (0xB002) to get gain map size
                        for i in range(ifd_count):
                            entry_offset = ifd_start + (i * 12)
                            if entry_offset + 12 <= len(mpf_data):
                                tag = struct.unpack(
                                    ">H", mpf_data[entry_offset : entry_offset + 2]
                                )[0]
                                if tag == 0xB002:  # MP Entry tag
                                    # Get offset to MP Entry data
                                    mp_entry_offset = struct.unpack(
                                        ">I",
                                        mpf_data[entry_offset + 8 : entry_offset + 12],
                                    )[0]
                                    # MP Entry data is relative to start of TIFF data (after MPF signature)
                                    if mp_entry_offset < len(
                                        mpf_data
                                    ) and mp_entry_offset + 32 <= len(mpf_data):
                                        # Skip first entry (16 bytes) and read second entry size
                                        second_entry_start = mp_entry_offset + 16
                                        if second_entry_start + 8 <= len(mpf_data):
                                            gain_map_size = struct.unpack(
                                                ">I",
                                                mpf_data[
                                                    second_entry_start
                                                    + 4 : second_entry_start
                                                    + 8
                                                ],
                                            )[0]
                                            info["gain_map_size"] = gain_map_size
                                    break

            pos += length

    except Exception as e:
        info["error"] = str(e)

    return info


def extract_relevant_exif(raw_path: str) -> bytes:
    """Extract relevant EXIF data from the original raw file.

    Args:
        raw_path: Path to the original raw file

    Returns:
        EXIF bytes ready for inclusion in JPEG
    """
    try:
        # Try to process the raw file and extract EXIF from the result
        with rawpy.imread(raw_path) as raw:
            # Create a temporary processed image to extract EXIF from
            temp_data = raw.postprocess(
                gamma=(1, 1),
                use_camera_wb=True,
                no_auto_bright=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
            )

            # Convert to PIL Image
            temp_pil = Image.fromarray(temp_data, mode="RGB")

            # Try to get EXIF from the processed image
            if hasattr(temp_pil, "getexif"):
                exif_dict = temp_pil.getexif()

                if exif_dict:
                    # Filter to preserve only relevant tags
                    preserve_tags = {
                        0x010F,  # Make
                        0x0110,  # Model
                        0xA434,  # LensModel
                        0x920A,  # FocalLength
                        0x829D,  # FNumber
                        0x829A,  # ExposureTime
                        0x8827,  # ISOSpeedRatings
                        0x0132,  # DateTime
                        0x9003,  # DateTimeOriginal
                        0x9004,  # DateTimeDigitized
                        0xA405,  # FocalLengthIn35mmFilm
                        0x8822,  # ExposureProgram
                        0x9207,  # MeteringMode
                        0x9208,  # LightSource
                        0x9209,  # Flash
                        0xA402,  # ExposureMode
                        0xA406,  # SceneCaptureType
                    }

                    # Create filtered EXIF dict
                    filtered_exif = {}
                    for tag, value in exif_dict.items():
                        if tag in preserve_tags:
                            filtered_exif[tag] = value

                    if filtered_exif:
                        # Create new EXIF object with filtered data
                        from PIL.ExifTags import TAGS

                        new_exif = Image.Exif()
                        for tag, value in filtered_exif.items():
                            try:
                                new_exif[tag] = value
                            except:
                                pass  # Skip problematic tags

                        # Convert to bytes
                        exif_bytes = new_exif.tobytes()

                        click.echo(
                            f"Extracted {len(filtered_exif)} EXIF tags from original file"
                        )

                        # Print extracted EXIF for debugging
                        click.echo("EXIF data found:")
                        for tag, value in filtered_exif.items():
                            tag_name = TAGS.get(tag, hex(tag))
                            click.echo(f"  {tag_name}: {value}")

                        return exif_bytes

    except Exception as e:
        click.echo(f"Warning: Could not extract EXIF from {raw_path}: {e}")

    click.echo("No EXIF data extracted")
    return b""


@click.command()
@click.version_option(version=__version__)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--verify", is_flag=True, help="Verify UltraHDR JPEG file instead of processing"
)
def cli(input_file, output, verify):
    """raw2ultra - A Python CLI tool for raw2ultra processing using native Python implementation."""

    if verify:
        # Verify mode - check if the input file is a valid UltraHDR JPEG
        click.echo(f"raw2ultra v{__version__}")
        click.echo(f"Verifying UltraHDR JPEG: {input_file}")

        info = verify_ultra_hdr_jpeg(input_file)

        click.echo(f"File size: {info['file_size']:,} bytes")
        click.echo(f"Is UltraHDR: {info['is_ultra_hdr']}")
        click.echo(f"Has XMP metadata: {info['has_xmp_metadata']}")
        click.echo(f"Has MPF structure: {info['has_mpf_structure']}")
        click.echo(f"Number of images: {info['num_images']}")
        click.echo(f"Gain map size: {info['gain_map_size']:,} bytes")

        if "error" in info:
            click.echo(f"Error: {info['error']}")

        if info["is_ultra_hdr"]:
            click.echo("✅ File appears to be a valid UltraHDR JPEG")
        else:
            click.echo("❌ File does not appear to be a valid UltraHDR JPEG")

        return

    if output:
        output = output + ".jpg"
    else:
        output = input_file + ".jpg"

    click.echo(f"raw2ultra v{__version__}")
    click.echo(f"Processing file: {input_file}")
    click.echo(f"Output file: {output}")

    click.echo("=== Processing HDR Intent (Linear) ===")

    # Process HDR intent with linear transfer function
    with rawpy.imread(input_file) as raw:
        # ndarray of shape (h,w,c) - keep linear for processing
        hdr_data: np.ndarray = raw.postprocess(
            gamma=(1, 1),  # Linear gamma
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.P3D65,  # P3 color space
            output_bps=16,
        )

    # Convert to linear float data (0-1+ range)
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

    click.echo("=== Creating SDR and Gain Map ===")

    # Create SDR version through tone mapping
    sdr_srgb_data = tone_map_to_sdr(hdr_linear_data)

    # Calculate gain map
    gain_map_data = calculate_gain_map(hdr_linear_data, sdr_srgb_data)

    # Convert to 8-bit for JPEG encoding
    sdr_8bit = (sdr_srgb_data * 255).astype(np.uint8)
    gain_map_8bit = (gain_map_data * 255).astype(np.uint8)

    click.echo("=== Creating UltraHDR JPEG with Native Python ===")

    # Extract EXIF data from original raw file
    click.echo("=== Extracting EXIF from Original File ===")
    original_exif = extract_relevant_exif(input_file)

    # Create JPEG data for SDR and gain map
    sdr_pil = Image.fromarray(sdr_8bit, mode="RGB")
    gain_map_pil = Image.fromarray(gain_map_8bit, mode="RGB")

    # Save SDR to bytes
    sdr_buffer = io.BytesIO()
    sdr_pil.save(sdr_buffer, "JPEG", quality=95, optimize=True)
    sdr_bytes = sdr_buffer.getvalue()

    # Save gain map to bytes
    gain_map_buffer = io.BytesIO()
    gain_map_pil.save(gain_map_buffer, "JPEG", quality=95, optimize=True)
    gain_map_bytes = gain_map_buffer.getvalue()

    try:
        # Create UltraHDR JPEG using native Python implementation
        ultra_hdr_data = create_ultra_hdr_jpeg(
            sdr_bytes, gain_map_bytes, width, height, original_exif
        )

        # Write the result
        with open(output, "wb") as f:
            f.write(ultra_hdr_data)

        click.echo(f"Successfully processed {input_file} and saved to {output}")
        click.echo(f"Output file size: {len(ultra_hdr_data)} bytes")

    except Exception as e:
        click.echo(f"Error processing {input_file}: {str(e)}")
        raise


if __name__ == "__main__":
    cli()
