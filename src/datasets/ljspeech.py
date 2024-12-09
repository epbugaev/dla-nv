import json
from pathlib import Path
from typing import Any, Literal

import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class LJSpeechDataset(BaseDataset):
    """
    LJSpeech dataset.

    Contains audios of different speakers.
    """
    def __init__(
        self,
        data_dir: str | None = None,
        index_dir: str | None = None, 
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            part (str): partition part
        """

        self.data_dir = (
            ROOT_PATH / "data" / "dla_dataset" if data_dir is None else Path(data_dir)
        )
        self.index_dir = (
            self.data_dir if index_dir is None else index_dir
        )

        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    @staticmethod
    def load_files(path: Path) -> tuple[Path, int]:
        info = torchaudio.info(path)
        lenght = info.num_frames / info.sample_rate
        return path, lenght

    def _get_or_load_index(self) -> list[dict[str, Any]]:
        suffix = "ljspeech_index.json"
        index_path = self.index_dir / suffix

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self) -> list[dict[str, str]]:
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            part (str): partition part
            load_video (bool): load video part or not
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        audio_data_path = (
            self.data_dir
        )

        assert (
            audio_data_path.exists() and audio_data_path.is_dir()
        ), f"No {audio_data_path} found!"

        print(f"Loading LJSpeech dataset")

        for mix_path in tqdm((audio_data_path).iterdir()):
            dataset_item = {}

            item_id = mix_path.name
            path, lenght = self.load_files(mix_path)
            dataset_item.update(
                {
                    "path": str(path),
                    "audio_len": lenght,
                    "id": item_id.rstrip(".wav"),
                }
            )

            index.append(dataset_item)

        print(index)
        return index