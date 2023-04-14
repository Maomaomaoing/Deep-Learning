import onnx
import numpy as np
import onnxruntime as rt
## onnxrutime (cpu), not use onnxruntime-gpu
from onnx import helper, numpy_helper
from onnx import TensorProto
import cv2
import time
import os

TRANS_CHANNEL_DIM = True
VALUE_SCALE_0_1 = True
BGR2RGB = True

#%%
## load model
model_file = "yolov3.onnx"
# image_file = "dog.jpeg"
image_file = "BMP_W768H576_dog.bmp"
so = rt.SessionOptions()
so.log_severity_level = 3
sess = rt.InferenceSession(model_file, so)

## Get the input name of the model
input_name = sess.get_inputs()[0].name
output_name = ["815","662","738","814"]
print(input_name, output_name)

img = cv2.imread(image_file)
img = cv2.resize(img, (416, 416))
## add batch dim
img = np.expand_dims(img, 0)
print("input shape: ", img.shape)
## BGR to RGB
if BGR2RGB:
    img = img[:, :, ::-1]
## 0~255 to 0~1.0
if VALUE_SCALE_0_1:
    img = img.astype(np.float32)
    img /= 255 
## bwhc to bcwh
if TRANS_CHANNEL_DIM:
    img = np.transpose(img, [0, 3, 1, 2]) 
    print("input shape: ", img.shape)

out = sess.run(output_name, {input_name: img})

print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
## conf score > 0.9
res_id = np.where(out[0][0,:,4]>0.9)[0]
for i in res_id:
    ## class score > 0.9
    class_id = np.where(out[0][0,i,5:]>0.9)[0]
    for j in class_id:
        print("class:", class_id, ", box:", out[0][0,i,:4])
        print("class score=", out[0][0,i,j+5])
