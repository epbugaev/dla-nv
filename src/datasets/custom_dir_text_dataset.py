from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset

class CustomDirTextDataset(BaseDataset):
    def __init__(self, audio_dir, *args, **kwargs):
        """
        Dataset to read from a folder of text files.

        Args:
            audio_dir (str or Path): Path to directory containing text files
        
        The dataset will load all text files with extension .txt from audio_dir.
        For each text file, it creates an entry with:
            - path: Full path to the text file
            - text: Contents of the text file
            - audio_len: Set to 1 (placeholder value since no audio is used)
            - id: Filename (without extension)
        """
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".txt"]:
                entry["path"] = str(path)
                                    
                transc_path = Path(entry["path"])
                if transc_path.exists():
                    with transc_path.open() as f:
                        entry["text"] = f.read().strip()
                
                entry["audio_len"] = 1 # Fill with fake info, it is not used when there are no original audio scripts
                entry["id"] = path.stem
                
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)