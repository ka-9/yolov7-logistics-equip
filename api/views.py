from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import onnxruntime as rt
from preprocess import preprocess

app = FastAPI()

MODEL_PATH = '../yolov7/runs/train/yolo_bmw_det3/weights/best.onnx'

# Load the model
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

@app.get('/')
async def root():
    return {'available_models': 'Hi'}

@app.get('/models')
async def get_available_models():
    return {'models': 'Hi'}


@app.post('/infer')
async def inference(image: UploadFile = File(...)):
    try:
        # Read input image
        contents = await image.read()
        img_data = preprocess(contents)

        # Perform inference
        outputs = sess.run([output_name], {input_name: img_data }) 
        print(outputs)

        # Construct JSON response with bounding boxes
        response = {'output': outputs}

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/infer-overlay')
async def inference_with_overlay(image: UploadFile = File(...)):
    try:
        # Read input image
        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform inference

        # Overlay bounding boxes on the input image
        # Replace this part with your actual inference and overlay logic

        # Encode the image with bounding boxes overlaid to return it as a response
        _, img_encoded = cv2.imencode('.jpg', image_np)
        return img_encoded.tobytes()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn views:app --reload
