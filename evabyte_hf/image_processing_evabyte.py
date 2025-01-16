# coding=utf-8
"""Image processor class for EvaByte."""

from typing import Dict, List, Optional, Union, Tuple

import io
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import (
    ImageInput,
    PILImageResampling,
    valid_images,
    validate_preprocess_arguments,
)
from PIL import Image

def _get_qtable_bytes():
    return {
        5: b'\xff\xd8\xff\xdb\x00C\x00\xa0nx\x8cxd\xa0\x8c\x82\x8c\xb4\xaa\xa0\xbe\xf0\xff\xff\xf0\xdc\xdc\xf0\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xdb\x00C\x01\xa0\xb4\xb4\xf0\xd2\xf0\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xd9',
        10: b'\xff\xd8\xff\xdb\x00C\x00P7<F<2PFAFZUP_x\xc8\x82xnnx\xf5\xaf\xb9\x91\xc8\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xdb\x00C\x01PZZxix\xeb\x82\x82\xeb\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xd9',
        15: b'\xff\xd8\xff\xdb\x00C\x005%(/(!5/+/<95?P\x85WPIIP\xa3u{a\x85\xc1\xaa\xcb\xc8\xbe\xaa\xba\xb7\xd5\xf0\xff\xff\xd5\xe2\xff\xe6\xb7\xba\xff\xff\xff\xff\xff\xff\xff\xff\xff\xce\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xdb\x00C\x015<<PFP\x9dWW\x9d\xff\xdc\xba\xdc\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xd9',
        20: b'\xff\xd8\xff\xdb\x00C\x00(\x1c\x1e#\x1e\x19(#!#-+(0<dA<77<{X]Id\x91\x80\x99\x96\x8f\x80\x8c\x8a\xa0\xb4\xe6\xc3\xa0\xaa\xda\xad\x8a\x8c\xc8\xff\xcb\xda\xee\xf5\xff\xff\xff\x9b\xc1\xff\xff\xff\xfa\xff\xe6\xfd\xff\xf8\xff\xdb\x00C\x01(--<5<vAAv\xf8\xa5\x8c\xa5\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xff\xd9',
        25: b'\xff\xd8\xff\xdb\x00C\x00 \x16\x18\x1c\x18\x14 \x1c\x1a\x1c$" &0P40,,0bFJ:Ptfzxrfpn\x80\x90\xb8\x9c\x80\x88\xae\x8anp\xa0\xda\xa2\xae\xbe\xc4\xce\xd0\xce|\x9a\xe2\xf2\xe0\xc8\xf0\xb8\xca\xce\xc6\xff\xdb\x00C\x01 $$0*0^44^\xc6\x84p\x84\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xff\xd9',
        30: b'\xff\xd8\xff\xdb\x00C\x00\x1b\x12\x14\x17\x14\x11\x1b\x17\x16\x17\x1e\x1c\x1b (B+(%%(Q:=0B`Ued_U][jx\x99\x81jq\x90s[]\x85\xb5\x86\x90\x9e\xa3\xab\xad\xabg\x80\xbc\xc9\xba\xa6\xc7\x99\xa8\xab\xa4\xff\xdb\x00C\x01\x1b\x1e\x1e(#(N++N\xa4n]n\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xff\xd9',
        50: b'\xff\xd8\xff\xdb\x00C\x00\x10\x0b\x0c\x0e\x0c\n\x10\x0e\r\x0e\x12\x11\x10\x13\x18(\x1a\x18\x16\x16\x181#%\x1d(:3=<9387@H\\N@DWE78PmQW_bghg>Mqypdx\\egc\xff\xdb\x00C\x01\x10\x12\x12\x18\x15\x18/\x1a\x1a/cB8Bcccccccccccccccccccccccccccccccccccccccccccccccccc\xff\xd9',
        75: b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xdb\x00C\x01\x08\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222222222222222222222222222222222222222222222222\xff\xd9',
        95: b'\xff\xd8\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff\xd9',
        100: b'\xff\xd8\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xd9',
    }


def _resize_if_exceeding_max_len(
    width: int, height: int, min_len: Optional[int] = 16, max_len: Optional[int] = None
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        height (`int`):
            Height of the input image.
        width (`int`):
            Width of the input image.
        max_len (`Dict[str, int]`, *optional*, defaults to the maximum size of the image):
            Defines the maximum dimensions of the image.

    Returns:
        The output size of the image after resizing.
    """
    max_len = max(height, width) if max_len is None else max_len
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1

    # Avoid resizing to a size smaller than 1
    height = max(height, min_len)
    width = max(width, min_len)
    return width, height

class EvaByteImageProcessor(BaseImageProcessor):

    model_input_names = []
    
    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        size: Dict[str, int] = None,
        do_convert_rgb: bool = True,
        jpeg_quality: int = 25,
        jpeg_subsampling: str = "4:2:0",
        jpeg_streamtype: str = 2,
        jpeg_restart_marker_blocks: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.size = size if size is not None else {"longest_edge": 384}
        self.do_convert_rgb = do_convert_rgb
        self.jpeg_quality = jpeg_quality
        self.jpeg_subsampling = jpeg_subsampling
        self.jpeg_streamtype = jpeg_streamtype
        self.jpeg_restart_marker_blocks = jpeg_restart_marker_blocks

    def jpeg_encode(
        self, 
        image,
        jpeg_quality,
        jpeg_subsampling,
        jpeg_streamtype,
        jpeg_restart_marker_blocks,
    ):
        with io.BytesIO() as output:
            image.save(
                output, 
                format="JPEG",
                quality=jpeg_quality, 
                subsampling=jpeg_subsampling, 
                streamtype=jpeg_streamtype, 
                restart_marker_blocks=jpeg_restart_marker_blocks
            )
            jpeg_bytes = output.getvalue()
        return jpeg_bytes

    def jpeg_merge_qtables(
        self,
        image_bytes,
        jpeg_quality=None,
    ):
        if jpeg_quality is None:
            jpeg_quality = self.jpeg_quality
        qtable_bytes = _get_qtable_bytes()[jpeg_quality]
        return image_bytes[:2] + qtable_bytes[2:-2] + image_bytes[2:]
        
    def resize(
        self,
        image: Image,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.LANCZOS,
    ) -> Image:
        if "longest_edge" in size:
            width, height = image.size
            # Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
            width, height = _resize_if_exceeding_max_len(width, height, max_len=size["longest_edge"])
            size = (width, height)
        elif "width" in size and "height" in size:
            size = (size["width"], size["height"])
        else:
            raise ValueError("size must be a dictionary with key 'longest_edge' or 'height' and 'width'.")
        resized_image = image.resize(size, resample=resample)
        return resized_image

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        resample = None,
        size: Dict[str, int] = None,
        do_convert_rgb: bool = None,
        jpeg_quality: int = None,
        jpeg_subsampling: str = None,
        jpeg_streamtype: str = None,
        jpeg_restart_marker_blocks: int = None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        jpeg_quality = jpeg_quality if jpeg_quality is not None else self.jpeg_quality
        jpeg_subsampling = jpeg_subsampling if jpeg_subsampling is not None else self.jpeg_subsampling
        jpeg_streamtype = jpeg_streamtype if jpeg_streamtype is not None else self.jpeg_streamtype
        jpeg_restart_marker_blocks = jpeg_restart_marker_blocks if jpeg_restart_marker_blocks is not None else self.jpeg_restart_marker_blocks

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        images_list = images
        if do_convert_rgb:
            images_list = [
                [
                    image.convert("RGB") for image in images
                ]
                for images in images_list
            ]

        if do_resize:
            images_list = [
                [
                    self.resize(image=image, size=size, resample=resample)
                    for image in images
                ]
                for images in images_list
            ]

        jpeg_bytes = [
            [
                self.jpeg_encode(
                    image,
                    jpeg_quality,
                    jpeg_subsampling,
                    jpeg_streamtype,
                    jpeg_restart_marker_blocks
                ) for image in images
            ]
            for images in images_list
        ]
        return jpeg_bytes
