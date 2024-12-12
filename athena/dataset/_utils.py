import abc
import pickle
from abc import ABC
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib import request

from anndata import AnnData
from typing import Union

PathLike = Union[str, Path]

@dataclass
class DataSet(ABC):

    name: str
    url: str

    doc_header: str = field(default=None, repr=False)
    path: PathLike = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.path is None:
            object.__setattr__(self, "path", os.path.expanduser(f"~/.cache/athena/{self.name}"))

    @property
    def _extension(self) -> str:
        return '.pkl'

    def __call__(self, path: PathLike = None, force_download: bool = False):
        return self.load(path, force_download)

    def load(self, fpath: PathLike = None, force_download: bool = False):
        """Download dataset form url"""
        fpath = str(self.path if fpath is None else fpath)

        if not fpath.endswith(self._extension):
            fpath += self._extension

        if not os.path.isfile(fpath) or force_download:
            # download file
            dirname = Path(fpath).parent
            if not dirname.is_dir():
                dirname.mkdir(parents=True, exist_ok=True)

            self._download_progress(Path(fpath), self.url)

        with open(fpath, 'rb') as f:
            ad = pickle.load(f)
        return ad

    def _download_progress(self, fpath: Path, url):
        from tqdm import tqdm
        from urllib.request import urlopen, Request

        blocksize = 1024 * 8
        blocknum = 0

        try:
            with urlopen(Request(url, headers={"User-agent":"ATHENA-user"})) as rsp:
                total = rsp.info().get("content-length", None)
                with tqdm(
                        unit="B",
                        unit_scale=True,
                        miniters=1,
                        unit_divisor=1024,
                        total=total if total is None else int(total)
                ) as t, fpath.open('wb') as f:
                    block = rsp.read(blocksize)
                    while block:
                        f.write(block)
                        blocknum += 1
                        t.update(len(block))
                        block = rsp.read(blocksize)
        except (KeyboardInterrupt, Exception):
            # Make sure file doesn’t exist half-downloaded
            if fpath.is_file():
                fpath.unlink()
            raise


    def _download(self, fpath: Path, url) -> None:
        try:
            path, rsp = request.urlretrieve(url, fpath)
        except (KeyboardInterrupt, Exception):
            # Make sure file doesn’t exist half-downloaded
            if path.is_file():
                path.unlink()
            raise

