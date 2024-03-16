# yolov7-logistics-equip

A YOLOv7 Computer Vision Model with Inference API that:
- Preprocesses Synthetic Data (Images and Labels)
- Fine-tunes a YOLOv7 model on Custom Synthetic Data
- Classifes material handling equipment (jacks, dollies, and bins)
- Working on Dockerizing the ONNX model API 

# Procedure

- This repo focused on 2 approaches, the first one was fine-tuning the YOLOv7 model and the second one had the goal of implementing the YOLOv3 model from scratch and training it.
- The purpose of this repo is only for practice. The code in the repo offers a solid foundation for CNN and computer vision models, fine tuning them on custom, preprocessed data and inferring from them.
- The data files were not included in this repo due to their size. The data was stored as follows:

BMW_data

├── train.csv  
├── test.csv  
├── images  
│   ├── train  
│   ├── test  
│   └── val  
└── labels  
  ├── train  
  ├── test  
  └── val  

---

# Data Pipeline

## Data Formatting: JSON to YOLO

- In order to train our YOLOv7 model, we need to convert the data from json to txt YOLO format
- The yolo format is as follows: class_label x_center y_center width height
- This has been handled in the `data_pipeline.ipynb` notebook.
- Additionally, the data has been visualized for ground truth reference.
  ![image](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/177347be-e310-47ef-bebe-87a2ddbed948)
  ![image](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/dfc91345-0b3d-4d36-b0e2-3a7b3961066d)

## Data bboxes Clamping

- In some instances, noisy data induced some of the normalized bounding box dimensions to be larger than 1. This means that in the original data, the coordinates for the bounding boxes were larger than the size / boundaries of the said image.
- To address this issue, a condition was added in the codee to clamp the data points with noisy data, since they could affect the process during data augmentation and model training.
- The faulty data accounted for < 5% of the overall data (136 instances).

## Data Augmentation: Albumentations

- The data was then augmented using an [Albumentations](https://albumentations.ai/docs/getting_started/) transformation pipeline
  ![image](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/87b7522b-6eed-40d0-a686-dcd9e21d648a)

## Train Test Val Split

- After formatting our data and augmenting it, we split the data into train and val
- Although some references suggest not to include augmented data in validation, in our scenario augmented data can help the model better by exposing it to a wider variety of examples during validation
- The train val split was handled in `data_pipeline.ipynb`
- The split ratio was chosen to be 20% of training data for validation

## Data IDs reformatting

- While cleaning the data, we realized that the class labels were:
  1. **unordered**, which may cause issues when encoding while training the model, since the model might infer that there are other classes in between. Therefore, additional classes wuld have to be passed to the model which is inefficient and unnecessary, and could lead to inconveniences.
  2. **different classes between train (and val) and test**. In fact, the classes 4, 5 and 7 were used for `dollies`, `bins` and `jacks` respectively; while in test, the labels were 4, 11 and 9 respectively.
- To address this issue, a method parsing the entire txt dataset insured that the classes for train and test were reformatted as follows:
  - `dollies`: 0
  - `bins`: 1
  - `jacks`: 2
- This insured data consistency in the whole dataset.

---

# YOLOv7 Training

## Preparing for Training

- The first step was cloning the repo [https://github.com/WongKinYiu/yolov7/](https://github.com/WongKinYiu/yolov7/)
- `cd /yolov7` and `pip install requirements.txt`
- In addition, we downloaded pretrained yolov7 frozen weights in order to perform fine tuning on top of them
- We made sure to create the `bmw_data.yaml` file for the yolov7 model to look for. This file will be the checkpoint and reference for the location of train, test, val for images and labels.
- To train the model: `!python train.py --img-size 640 --cfg cfg/training/yolov7.yaml --hyp data/hyp.scratch.custom.yaml --batch 4 --epochs 50 --data data/bmw_data.yaml --weights yolov7.pt --workers 24 --name yolo_bmw_det2`
- Training the model on images of size 3 * 640 * 640, configuration yolov7.yaml (default), custom hyperparameter that will later be tuned, 50 epochs
- The new weights will be saved under the name `yolo_bmw_det2` in the `runs` folder

## Training the model

- While training the model, it is important to always monitor the performance and evolution of critical parameters such as precision, recall, mAP score, loss.
- Tensorboard was used in order to monitor these parameters, as shown below:

![Screenshot 2024-03-14 002629](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/f89e3253-fde5-430c-8d81-5d16401671ba)

- Moreover, training was performed locally using Nvidia GEForce RTX 2060

![Screenshot 2024-03-14 013731](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/dd685f54-a04c-46d7-9604-0cc434e3edb3)

- After training for 50 epochs, the loss functions as well as precision recall and f1 scores all indicated that a plateau was reached. This could hint at overfitting, as well as unnecessary computing beyond which accurarcy stays the same.
- After hyperparameter tuning, most notably reducing epochs to 30 instead of 50, training was performed again

![Screenshot 2024-03-14 120604](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/42c850c8-1fb8-44d0-a53f-0b77ce1da4d0)

- After training was finished, we obtained the results below:

<div style="display:flex; flex-wrap:wrap;">
    <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/F1_curve.png" alt="img" style="width:45%; margin:5px;">
    <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/P_curve.png" alt="img" style="width:45%; margin:5px;">
    <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/R_curve.png" alt="img" style="width:45%; margin:5px;">
    <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/results.png" alt="img" style="width:45%; margin:5px;">
</div>

## Inference Test

<div style="display:flex; flex-wrap:wrap;">
    <div style="width:45%; margin:5px;">
        <h3>Ground truth</h3>
        <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/test_batch2_labels.jpg" alt="img" style="width:100%;">
    </div>
    <div style="width:45%; margin:5px;">
        <h3>Prediction</h3>
        <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/test_batch2_pred.jpg" alt="img" style="width:100%;">
    </div>
    <div style="width:45%; margin:5px;">
        <h3>Ground truth</h3>
        <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/test_batch0_labels.jpg" alt="img" style="width:100%;">
    </div>
    <div style="width:45%; margin:5px;">
        <h3>Prediction</h3>
        <img src="https://github.com/ka-9/yolov7-logistics-equip/blob/main/yolov7/runs/train/yolo_bmw_det3/test_batch0_pred.jpg" alt="img" style="width:100%;">
    </div>
</div>


# ONNX Export

- The `yolo_bmw_det3` model weights were exported as onnx: `!python export.py --weights runs/train/yolo_bmw_det3/weights/best.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640`

# Netro Visualization

- Netro .svg and .png graphs can be found [here](https://github.com/ka-9/yolov7-logistics-equip/blob/main/netron/best.onnx.png)

---

# RetinaNet inspired model

- The purpose of working on this model was to acquire a deeper understanding of the fundementals of backbone transfer learning.
- ResNet acts as a new **backbone network** architecture, which is larger and more accurate than starting from scratch.
  Below are some figures from the training phase:

![Figure_2](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/c057c5a6-5dea-4f02-8686-29a64a9533fc)
![Figure_1](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/993fad8c-2bc8-4b49-9f22-b20269cb65b3)
![Figure_3](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/304098c7-be98-4094-98eb-0aa82d6f3c54)

---

# API and ONNX Inference Session

- In order to take this project to the next level, we implemented a FastAPI interfacing framework that interacts with onnx inferrence sessions in order to provide a more convenient and scalable way to interact with our developed models.
- Even though the APIs are under development, they are a proof of scalability and consitute a stable foundation for later improvements.
- To activate runtime, navigate to api folder: `cd ./api` from root dir
- Run the command: `uvicorn views:app --reload`
- Open the localhost port that shows on the terminal, ex. http://127.0.0.1:8000, followed by /docs

![image](https://github.com/ka-9/yolov7-logistics-equip/assets/99538511/0e488598-ad4d-4eb2-aa98-fb1aa9e2f6b4)
