import io
import json
import re
import sys
from pathlib import Path
import zstandard as zstd
import tqdm


def format_category(category):
    return re.sub(r'[^a-z0-9_]', '', category.lower().replace('-', '_'))


def process_file(input_file_path):
    base_file = Path(input_file_path).stem
    ctx = zstd.ZstdDecompressor()
    compressors = {}

    with open(input_file_path, 'rb') as compressed_file:
        with ctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in tqdm.tqdm(text_stream):
                if not line.strip():
                    continue  # Skip empty lines

                # Decode line to string and load as JSON
                data = json.loads(line)
                category = data['meta']['pile_set_name']
                category = format_category(category)
                output_dir = Path(category)
                output_dir.mkdir(exist_ok=True)
                output_file_path = output_dir / f"{base_file}.zst"

                # Check if compressor exists for this category, if not create it
                if category not in compressors:
                    output_file = open(output_file_path, 'wb')
                    compressors[category] = zstd.ZstdCompressor().stream_writer(output_file)

                # Write to the compressor
                compressors[category].write(line.encode('utf-8'))
                compressors[category].flush()

    # Close all open compressors
    for compressor in compressors.values():
        compressor.close()


if __name__ == '__main__':
    for path in sys.argv[1:]:
        process_file(path)