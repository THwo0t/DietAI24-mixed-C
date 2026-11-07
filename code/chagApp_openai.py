from openai import OpenAI
import base64
import io
import os
from PIL import Image
from api_log import openai_client
from config import temperature_for, completion_limit

# paper.txt:237-240: every image is standardized to 256 x 256 pixels, nothing else is changed
IMAGE_SIZE = (256, 256)


def open_img(image_path):
    # return a data url with base64 encoding
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return "data:image/png;base64," + encoded_string.decode('utf-8')


def resize_img(image_path):
    """Data URL of the image resized to 256 x 256 (bicubic, aspect ratio not kept), as PNG.

    ASA24 images keep their transparent background, as the authors sent them. Colour and
    transparency are resized separately: resizing RGBA in one go premultiplies the alpha and
    turns the white under the transparent pixels black.
    """
    with Image.open(image_path) as image:
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGBA')
        resized = image.convert('RGB').resize(IMAGE_SIZE, Image.BICUBIC)
        if image.mode == 'RGBA':
            resized.putalpha(image.getchannel('A').resize(IMAGE_SIZE, Image.BICUBIC))
        buffer = io.BytesIO()
        resized.save(buffer, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')


def image_url(link):
    """The image URL sent to the API: a local image file is resized and sent as a data URL."""
    return resize_img(link) if os.path.isfile(link) else link


class Vision:
    def __init__(self, model_name):
        self.client = openai_client()

        self.model_name = model_name
        self.messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing images with computer vision. I will present you with a picture of food, which might be placed on a plate, inside a spoon, or contained within different vessels. Your job is to accurately identify the food depicted in the image."
                # "content": "You are trained to interpret images about food and make responsible assumptions about them."
            }
        ]
        
    def chat(self, message, image_path):
        # Each call is independent: the system prompt plus this question and image only
        messages = self.messages + [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message,
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url(image_path)
                    },
                }
            ]
        }]
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature_for(self.model_name),
            **completion_limit(self.model_name, 600)
        )
        response = res.choices[0].message.content
        return response

    def append_message(self, message):
        self.messages.append(message)
