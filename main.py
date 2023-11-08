from fastapi import FastAPI, File, UploadFile
from typing import Annotated
import tensorflow as tf
from typing import Union
from PIL import Image
import io
import os
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/")
def predict(name: str, number: int):
	return {"nama": name, "nomor": number,"Status": "OK mantap"}

model = tf.keras.models.load_model('best_model_inceptionV3.h5')

@app.get("/class/")
def class_model():
	return {
    	'Bill Gates': 0,
 		'Chris Evans': 1,
 		'Cristiano Ronaldo': 2,
 		'Mark Zuckerberg': 3,
 		'Tom Holland': 4
   	}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((32, 32))
    image_array = np.array(image)
    image_array = image_array / 255.0
    
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
    
def predict_image(image_bytes):
    preprocessed_image = preprocess_image(image_bytes)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

def who(classNum):
    if classNum == 0:
        return 'Bill Gates'
    elif classNum == 1:
        return 'Chris Evans'
    elif classNum == 2:
        return 'Cristiano Ronaldo'
    elif classNum == 3:
        return 'Mark Zuckerberg'
    elif classNum == 4:
        return 'Tom Holland'
    else:
     	return 'Unknown'

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

# @app.post("/predict/")
# async def predict(file: UploadFile):
#     image_bytes = await file.read()
#     prediction = predict_image(image_bytes)
#     # return {"prediction": prediction.tolist()}
#     return {"prediction": file.size}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as bytes
        image_bytes = await file.read()
        # Perform prediction
        prediction_result = predict_image(image_bytes)
        result = who(prediction_result[0])
        return {"prediction": result}
    
    except Exception as e:
        return {"error": str(e)}