import os

from .reader_image_folder import ReaderImageFolder
from .reader_image_in_tar import ReaderImageInTar


def create_reader(root, depth_root, **kwargs):
    
    reader = ReaderImageFolder(root, depth_root, **kwargs)

    return reader
