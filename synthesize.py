import argparse
import os
from pathlib import Path
import tempfile

def main():
    parser = argparse.ArgumentParser(description='Run inference on text files in a folder or a single text string')
    parser.add_argument('--folder', type=str, help='Path to folder containing text files')
    parser.add_argument('--text', type=str, help='Text string to synthesize')
    args = parser.parse_args()

    if not args.folder and not args.text:
        raise ValueError("Either --folder or --text must be provided")
    if args.folder and args.text:
        raise ValueError("Cannot provide both --folder and --text")

    if args.folder:
        folder_path = Path(args.folder).absolute()
        if not folder_path.exists():
            raise ValueError(f"Folder {folder_path} does not exist")
        
        os.system(f'python inference.py -cn inference datasets.val.audio_dir={folder_path}')
    else:
        # Create temporary directory and text file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            text_file = temp_path / "input.txt"
            text_file.write_text(args.text)
            folder_path = temp_path

            os.system(f'python inference.py -cn inference_text_custom datasets.val.audio_dir={folder_path}')

if __name__ == '__main__':
    main()