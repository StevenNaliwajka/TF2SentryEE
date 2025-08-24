import re
import cv2

"""
This file will contain global (static) helper functions that will be used throughout the project.
this will be useful for reusing checks for CUDA support and potentially speeding up the execution by allowing for
GPU computations.
"""


def has_compiled_with_cuda() -> bool:
    return re.search(r'NVIDIA CUDA:\s*YES', cv2.getBuildInformation()) is not None
