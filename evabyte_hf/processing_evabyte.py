# coding=utf-8
"""
Processor class for EvaByte.
"""
import base64
from io import BytesIO

import requests
import os
import PIL
from PIL import Image

from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, to_py_obj

def fetch_image(image: Union[str, "PIL.Image.Image"]) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(BytesIO(requests.get(image, timeout=None).content))
    elif os.path.isfile(image):
        image_obj = Image.open(image)
    elif image.startswith("data:image/"):
        image = image.split(",")[1]
        # Try to load as base64
        try:
            b64 = base64.decodebytes(image.encode())
            image = PIL.Image.open(BytesIO(b64))
        except Exception as e:
            raise ValueError(
                f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
            )
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")

    return image_obj

def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")

def is_file(val) -> bool:
    return isinstance(val, str) and os.path.isfile(val)

def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem) or is_file(elem)

vl_chat_template = """
{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{{- '<|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>'}}

{%- for message in messages %}
    {%- if (message['role'] != 'user') and (message['role'] != 'assistant') %}
        {{- raise_exception('Conversation roles must be user or assistant') }}
    {%- endif %}
    
    {%- if message['content'] is string %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] + '<|eot_id|>' }}
    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
        {%- for content in message['content'] %}
            {%- if content['type'] == 'image' %}
                {{- '<image_placeholder>\n' }}
            {%- elif content['type'] == 'text' %}
                {{- content['text'] }}
            {%- endif %}
        {%- endfor %}
        {{- '<|eot_id|>' }}        
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
{%- endif %}
"""

class EvaByteProcessor(ProcessorMixin):
    r"""
    Constructs a EvaByte processor which wraps a EvaByte image processor and a EvaByte tokenizer into a single processor.

    [`EvaByteProcessor`] offers all the functionalities of [`EvaByteImageProcessor`] and [`EvaByteTokenizer`]. See the
    [`~EvaByteProcessor.__call__`] and [`~EvaByteProcessor.decode`] for more information.

    Args:
        image_processor ([`EvaByteImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`EvaByteTokenizer`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.t2v_token_id = self.tokenizer.convert_tokens_to_ids("<t2v_token>")
        self.v2t_token_id = self.tokenizer.convert_tokens_to_ids("<v2t_token>")
        self.image_placeholder = "<image_placeholder>"
        self.vl_chat_template = vl_chat_template

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        strip_ending_sentinel: bool = False,
        encode_only: bool = False,
        **kwargs
    ) -> Union[BatchFeature, List[List[int]]]:
        # processing pipeline:
        # 1. read images or videos from paths
        # 2. use image_processor to convert images / videos to byte streams
        if images is not None:
            if isinstance(images, bytes):
                image_bytes_list = [[images]]
            elif isinstance(images, list) and isinstance(images[0], bytes):
                image_bytes_list = [images]
            elif isinstance(images, list) and isinstance(images[0], list) and isinstance(images[0][0], bytes):
                image_bytes_list = images
            else:
                if is_image_or_image_url(images):
                    images = [[images]]
                elif isinstance(images, list) and is_image_or_image_url(images[0]):
                    images = [images]
                elif (
                    not isinstance(images, list)
                    and not isinstance(images[0], list)
                    and not is_image_or_image_url(images[0][0])
                ):
                    raise ValueError(
                        "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                    )
                # Load images if they are URLs
                images = [[fetch_image(im) if is_url(im) or is_file(im) else im for im in sample] for sample in images]
                image_bytes_list = self.image_processor(images=images, **kwargs)

        if not isinstance(text, list):
            text = [text]
        assert len(text) == 1, "Only support batch size 1 for now"
        assert len(text) == len(image_bytes_list), "text and image_bytes_list must have the same length"
        # TODO: invoke SequenceFeatureExtractor to get batched inputs

        # 3. tokenize the text and put images / videos byte streams into the placeholders
        #    surrounded by special tokens like "<image>" and "</image>"
        batch_input_ids = []
        if not encode_only:
            batch_attention_mask = []
        else:
            batch_attention_mask = None

        for t, image_bytes in zip(text, image_bytes_list):
            text_splits = t.split(self.image_placeholder)
            if len(text_splits) != len(image_bytes) + 1:
                raise ValueError(
                    f"The number of image tokens should be equal to the number of images, "
                    f"but got {len(text_splits)} and {len(image_bytes) + 1}"
                )

            input_ids = [self.tokenizer.bos_token_id]
            for i, text_part in enumerate(text_splits):
                # each text part must be non-empty because we added markers around placeholders
                split_tokens = self.tokenizer.encode(text_part, add_special_tokens=False)
                input_ids.extend(split_tokens)
                # Add image bytes after each text part except the last one
                if i < len(image_bytes):
                    input_ids.append(self.t2v_token_id)
                    input_ids.extend([b + self.tokenizer.offset for b in image_bytes[i]])
                    input_ids.append(self.v2t_token_id)

            if strip_ending_sentinel and (input_ids[-1] in [self.t2v_token_id, self.v2t_token_id]):
                input_ids = input_ids[:-1]

            batch_input_ids.append(input_ids)
            if not encode_only:
                batch_attention_mask.append([1] * len(input_ids))

        if not encode_only:
            # 4. return batch of features
            inputs = BatchFeature({
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask
            }, tensor_type=return_tensors)
            return inputs
            # # Pad sequences
            # padded_inputs = self.tokenizer.pad(
            #     {"input_ids": batch_input_ids},
            #     padding=True,
            #     return_attention_mask=True,
            #     return_tensors=return_tensors,
            # )
            # return BatchFeature(data=padded_inputs)
        else:
            return batch_input_ids

    def image_tokens_to_bytes(self, image_token_ids, jpeg_quality=None):
        image_bytes = bytes([token_id - self.tokenizer.offset for token_id in image_token_ids])
        image_bytes = self.image_processor.jpeg_merge_qtables(image_bytes, jpeg_quality)
        return image_bytes

    def batch_decode(self, sequences, **kwargs):
        """
        This method forwards all its arguments to EvaByteTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        rets = [self.decode(seq, **kwargs) for seq in sequences]
        return tuple(map(list, zip(*rets)))

    def decode(self, token_ids, **kwargs):
        """
        Decodes a sequence of input_ids, handling image tokens separately.
        Returns a tuple of (decoded_text, images), where images is a list of bytes.
        """
        if kwargs and "jpeg_quality" in kwargs:
            kwargs = kwargs.copy()
            jpeg_quality = kwargs.pop("jpeg_quality")
        else:
            jpeg_quality = None
        
        token_ids = to_py_obj(token_ids)
        # Find indices of t2v_token_id and v2t_token_id
        t2v_indices = [i for i, token_id in enumerate(token_ids) if token_id == self.t2v_token_id]
        v2t_indices = [i for i, token_id in enumerate(token_ids) if token_id == self.v2t_token_id]
        
        # Check for correct pairing of t2v and v2t tokens
        if len(t2v_indices) != len(v2t_indices):
            raise ValueError("Mismatched number of t2v and v2t tokens in token_ids: {} and {}".format(t2v_indices, v2t_indices))

        # Ensure t2v and v2t tokens are in the correct order
        for t2v_idx, v2t_idx in zip(t2v_indices, v2t_indices):
            if t2v_idx >= v2t_idx:
                raise ValueError("Found t2v_token_id after v2t_token_id in token_ids")

        # Initialize the start index
        images = []
        decoded_text = ""

        start = 0
        # Iterate over pairs of t2v and v2t indices
        for t2v_idx, v2t_idx in zip(t2v_indices, v2t_indices):
            # Decode text tokens before the image
            text_token_ids = token_ids[start:t2v_idx]
            if len(text_token_ids) > 0:
                decoded_text += self.tokenizer.decode(text_token_ids, **kwargs)

            # Insert image placeholder
            decoded_text += self.image_placeholder

            # Extract image tokens and convert them to bytes
            image_token_ids = token_ids[t2v_idx + 1 : v2t_idx]
            image_bytes = self.image_tokens_to_bytes(image_token_ids, jpeg_quality)
            images.append(image_bytes)

            # Update the start index to the token after v2t_token_id
            start = v2t_idx + 1

        # Decode any remaining text tokens after the last image
        if start < len(token_ids):
            text_token_ids = token_ids[start:]
            decoded_text += self.tokenizer.decode(text_token_ids, **kwargs)

        return decoded_text, images

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))