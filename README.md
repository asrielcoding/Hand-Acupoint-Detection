# Hand-Acupoint-Detection
This is a repository for Application of AI on Chinese Medicine: Acupoint Detection. It's the final project for the class and I will show steps that we have to take to finish this project.

- Data Collection
- Hand Detection
- Inpainting
- Keypoint Detection

## Data Collection
Taking picture of hands at different angles and different people(different skin color of hands) with a smart cellphone.
- Each angle for one minutes
- Five angles each hand
- Front and back of each hand
- Left and right hand per person
- One-second video can be divided into about 3600 frames by OpenCV

## Hand Detection
Use Yolov8 model for hand detection and cropping images bounded by bbox.
- Checkout data_process and dataprocess.ipynb for details and codes

## Inpainting
We annotated acupuncure points with blue dot stickers, and then convert BGR to HSV.
- Checkout dataprocess.ipynb and inpaint.py for details and codes