from pathlib import Path
import base64
import io

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow import keras


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True,
        )
        self.offset = self.add_weight(
            name="offset",
            shape=input_shape[-1:],
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "photo_to_sketch": BASE_DIR / "photo_to_sketch_generator.keras",
    "sketch_to_photo": BASE_DIR / "sketch_to_photo_generator.keras",
}

photo_to_sketch_model = None
sketch_to_photo_model = None
model_load_error = None


try:
    custom_objects = {"InstanceNormalization": InstanceNormalization}

    if not MODEL_PATHS["photo_to_sketch"].exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATHS['photo_to_sketch'].name}")
    if not MODEL_PATHS["sketch_to_photo"].exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATHS['sketch_to_photo'].name}")

    photo_to_sketch_model = keras.models.load_model(
        str(MODEL_PATHS["photo_to_sketch"]), custom_objects=custom_objects
    )
    sketch_to_photo_model = keras.models.load_model(
        str(MODEL_PATHS["sketch_to_photo"]), custom_objects=custom_objects
    )
    print("Models loaded successfully.")
except Exception as exc:
    model_load_error = str(exc)
    print(f"Model loading failed: {exc}")


def resize(image, height, width):
    return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def normalize(image):
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def preprocess_image(pil_image):
    image_array = np.array(pil_image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("The uploaded image must be a color image.")

    image_tensor = tf.convert_to_tensor(image_array)
    image_tensor = resize(image_tensor, 256, 256)
    image_tensor = normalize(image_tensor)
    return tf.expand_dims(image_tensor, 0)


@app.route("/", methods=["GET", "POST"])
def upload_and_translate():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template(
                "index.html", translated_image=None, error="Please choose an image file."
            )

        if model_load_error:
            return render_template(
                "index.html",
                translated_image=None,
                error=f"Model setup issue: {model_load_error}",
            )

        try:
            image_bytes = file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            hsv_image = np.array(pil_image.convert("HSV"))
            mean_saturation = float(np.mean(hsv_image[:, :, 1]))
            image_tensor = preprocess_image(pil_image)

            if mean_saturation < 25:
                prediction = sketch_to_photo_model(image_tensor, training=False)[0]
            else:
                prediction = photo_to_sketch_model(image_tensor, training=False)[0]

            prediction = tf.clip_by_value(prediction, 0.0, 1.0)
            prediction_denorm = (prediction.numpy() * 0.5 + 0.5)
            output_image = Image.fromarray((prediction_denorm * 255).astype(np.uint8))
            return render_template(
                "index.html",
                translated_image=encode_image(output_image),
                error=None,
            )
        except UnidentifiedImageError:
            return render_template(
                "index.html",
                translated_image=None,
                error="The uploaded file is not a valid image.",
            )
        except Exception as exc:
            return render_template(
                "index.html",
                translated_image=None,
                error=f"Could not process the image: {exc}",
            )

    return render_template("index.html", translated_image=None, error=None)


if __name__ == "__main__":
    app.run(debug=True)