# raw2ultra

A Python CLI tool for converting RAW images to UltraHDR JPEG format using **native Python implementation** - no external dependencies required!

## Features

- ✅ **Pure Python Implementation**: No need for external `ultrahdr_app` or other tools
- ✅ **RAW to UltraHDR**: Convert RAW images directly to UltraHDR JPEG format
- ✅ **Tone Mapping**: Advanced Reinhard tone mapping for SDR representation
- ✅ **Gain Map Generation**: Automatic calculation of HDR gain maps
- ✅ **MPF Structure**: Proper Multi-Picture Format embedding for compatibility
- ✅ **XMP Metadata**: Adobe-compatible gain map metadata
- ✅ **Verification**: Built-in tool to verify UltraHDR JPEG files
- ✅ **High Quality**: Optimized JPEG encoding with quality settings

## What is UltraHDR?

UltraHDR is Google's HDR image format that embeds a gain map into a standard JPEG file. This allows:

- **Backward Compatibility**: Regular JPEG viewers see the SDR image
- **HDR Enhancement**: HDR-capable displays show enhanced dynamic range
- **Single File**: Everything embedded in one JPEG file
- **Wide Support**: Works in Chrome, Android, and other HDR-capable platforms

## Installation

```bash
pip install -e .
```

## Usage

### Convert RAW to UltraHDR

```bash
# Basic conversion
raw2ultra image.arw

# Specify output filename
raw2ultra image.arw -o my_hdr_image

# The output will be my_hdr_image.jpg (extension added automatically)
```

### Verify UltraHDR Files

```bash
# Check if a JPEG file is a valid UltraHDR image
raw2ultra image.jpg --verify
```

Example verification output:
```
raw2ultra v0.1.0
Verifying UltraHDR JPEG: test_output.jpg
File size: 12,472,640 bytes
Is UltraHDR: True
Has XMP metadata: True
Has MPF structure: True
Number of images: 2
Gain map size: 3,446,044 bytes
✅ File appears to be a valid UltraHDR JPEG
```

## Technical Details

### Processing Pipeline

1. **RAW Processing**: Linear HDR data extraction using rawpy
2. **Tone Mapping**: Reinhard tone mapping for SDR representation
3. **Gain Map Calculation**: HDR/SDR ratio computation in log space
4. **JPEG Encoding**: High-quality JPEG compression for both images
5. **MPF Embedding**: Multi-Picture Format structure creation
6. **XMP Metadata**: Adobe-compatible gain map metadata injection

### Supported Formats

- **Input**: Any RAW format supported by rawpy (ARW, CR2, NEF, DNG, etc.)
- **Output**: UltraHDR JPEG with embedded gain map

### Image Processing

- **Color Space**: Display P3 for wide gamut support
- **Bit Depth**: 16-bit processing, 8-bit output
- **Scaling**: Automatic downscaling for images larger than 8192x8192
- **Dimension Alignment**: Automatic cropping to even dimensions

## Dependencies

- `rawpy`: RAW image processing
- `pillow`: JPEG encoding and image manipulation
- `numpy`: Numerical computations
- `scikit-image`: Image resizing and processing
- `click`: Command-line interface
- `imageio`: Image I/O operations

## How It Works

The tool implements the UltraHDR specification using pure Python:

1. **JPEG Parsing**: Custom JPEG segment parser
2. **MPF Structure**: Multi-Picture Format directory creation
3. **XMP Injection**: Adobe gain map metadata embedding
4. **Gain Map Encoding**: Logarithmic HDR/SDR ratio calculation

### Gain Map Formula

```
gain_linear = hdr_linear / sdr_linear
gain_log = log2(gain_linear)
gain_normalized = (gain_log + 4.0) / 8.0  # Normalize to 0-1 range
```

### Tone Mapping

Uses Reinhard tone mapping with luminance-based scaling:

```
L_out = L_in / (1 + L_in)
```

## Viewing UltraHDR Images

UltraHDR images can be viewed on:

- **Chrome Browser**: Full HDR support on compatible displays
- **Android Devices**: Native support in Android 14+
- **Google Photos**: HDR rendering on supported devices
- **Regular Viewers**: Fall back to SDR representation

## Comparison with External Tools

| Feature | raw2ultra (Native) | ultrahdr_app | libultrahdr |
|---------|-------------------|--------------|-------------|
| Dependencies | Python only | C++ binary | C++ library |
| Installation | pip install | Manual build | Manual build |
| Platform | Cross-platform | Platform specific | Platform specific |
| Integration | Native Python | Subprocess | FFI bindings |
| Customization | Full control | Limited | Full control |

## Development

### Building from Source

```bash
git clone <repository>
cd raw2ultra
pip install -e .
```

### Running Tests

```bash
# Test conversion
raw2ultra test_image.arw -o test_output

# Verify output
raw2ultra test_output.jpg --verify
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Google's UltraHDR specification
- Adobe's gain map technology
- The rawpy and PIL communities
