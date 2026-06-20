from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow import keras
from PIL import Image
import numpy as np
from imagePreprocess import preprocess
import io
import redis      # the Python library that lets your code talk to Redis database
import hashlib    # for creating the image hash for redis cache
import json       # for converting result to string for storage
import os         # for reading environment variables



app = FastAPI()    # Create the FastAPI app. This one line creates your entire web server


model = keras.models.load_model("digit_recognizer.h5", compile=False)


# Connect to Redis
# REDIS_HOST comes from docker-compose environment variable
# Falls back to 'localhost' when running without docker-compose
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=6379,
    db=0
)


# Written outside the function deliberately. Loads when server starts, stays in memory(RAM).
# Loading inside the endpoint/function would reload it on every request — very slow


# Endpoint
@app.get("/")
def root():
    return {"message": "Digit Recognition API is running"}


@app.post("/predict")  # This line connects a URL to a function. When someone hits that URL, that function runs.
async def predict(file: UploadFile = File(...)):   # file: UploadFile means the request must contain a file. File(...) means it's required — no file, no response.

    if not file.content_type.startswith("image/"):   # Every file has a content type — image/png, image/jpeg, application/pdf etc. If it doesn't start with image/, reject it immediately before doing anything else.
        raise HTTPException(status_code=400, detail="File must be an image")

    try:   # If the file is corrupted or unreadable, PIL will crash. The try/except catches that crash and returns a clean message instead.

        # Read the uploaded image as raw bytes into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        # io.BytesIO(contents) — wraps the raw bytes so PIL can read them like a file without saving to disk
        # Image.open(...) — opens it as a PIL image
        # .convert("L") — converts to grayscale. Your model was trained on grayscale — a colour image would confuse it

    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file. It may be corrupted.")

# Redis cache
    # Hash the image bytes to create a unique cache key
    image_hash = hashlib.md5(contents).hexdigest()  # Converts image bytes → 32 character unique string. Same image = same string every time.

    # Look up hash in Redis — if found, return cached result immediately
    cached_result = redis_client.get(image_hash)
    if cached_result:
        return json.loads(cached_result)  # return without running model


    # Preprocess the image(crops, pads, resizes to 28×28, normalizes, reshapes to (1, 784).Returns a numpy array ready for the model)
    img_array = preprocess(image)
    if img_array is None:  # preprocess() returns None when the image has no digit. Catch that here.

        raise HTTPException(status_code=400, detail="Image appears to be blank. Please send an image containing a digit.")

    try:   # If something unexpected breaks during prediction, the server returns a 500 error instead of crashing entirely.

        # Runs the image through the model.Returns 10 probabilities—one per digit. verbose=0 means don't print anything while predicting
        prediction = model.predict(img_array, verbose=0)

        # Get the digit with the highest probability
        digit = int(np.argmax(prediction))
        confidence = float(round(np.max(prediction) * 100, 1))

        # Get top 3 predictions
        top3_idx = np.argsort(prediction[0])[::-1][:3]
        top3 = [
            {
                "digit": int(i),
                "confidence": round(float(prediction[0][i] * 100), 1)
            }
            for i in top3_idx
        ]

        result = {
            "digit": digit,
            "confidence": confidence,
            "top3": top3
        }

        # Store result in Redis cache
        # 3600 seconds = 1 hour expiry
        redis_client.setex(image_hash, 3600, json.dumps(result))

        return result


    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed. Something went wrong on the server.")