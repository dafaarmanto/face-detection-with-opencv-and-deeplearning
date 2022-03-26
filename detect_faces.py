# Import the necessary libraries
import cv2
import numpy as np
import argparse

# Construct the argument parse and parse the argument

# --image: The path to the input image
# --prototxt: The path to the Caffe prototxt file
# --model: The path to the pre-trained Caffe model
# --confidence: Is optional argument, can overwrite the default threshold of 0.5 (as you wish)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 px and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
  # Extract the confidence associated with the prediction
  confidence = detections[0, 0, i, 2]

  # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
  if confidence > args["confidence"]:
    # Compute the (x, y)
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # draw the bounding box of the face along with probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)