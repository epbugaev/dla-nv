from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    """
    Dataset to read from a folder of wavs.
    """
    def __init__(self, audio_dir, *args, **kwargs):
        """
        Args:
            audio_dir (str or Path): Path to directory containing audio files
        
        The dataset will load all audio files with extensions .mp3, .wav, .flac, or .m4a from audio_dir.
        For each audio file, it creates an entry with:
            - path: Full path to the audio file
            - audio_len: Duration of the audio in seconds
            - id: Filename (without extension)
        """
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = length
                entry["id"] = path.stem
                
                print('entry:', entry)

            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)