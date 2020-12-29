import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Argument Parser
arg = argparse.ArgumentParser()
arg.add_argument("-i", "--image", type=str, required=True, help="path to input black and white image")
args = vars(arg.parse_args())

# model
proto = 'model/colorization_deploy_v2.prototxt'
weights = 'model/colorization_release_v2.caffemodel'

print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe(proto, weights)
pts_in_hull = np.load('model/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts_in_hull.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Reading Image
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Prediction
print("[INFO] colorizing image...")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

nam=args["image"]
nm = nam.split("/")

cv2.imwrite("output/colorize_"+nm[-1], colorized)

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)

cv2.waitKey(0)