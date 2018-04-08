# USAGE
# python test_network.py --model sketch_classification.model --image apple_test.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

names = ['airplane', 'alarm clock', 'angel', 'ant', 'apple']

# load the image
test_image = cv2.imread(args["image"])
orig = test_image.copy()

# pre-process the image for classification
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255

test_image= np.expand_dims(test_image, axis=3) 
test_image= np.expand_dims(test_image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
index = model.predict_classes(test_image)
result = names[index[0]]
prob = model.predict(test_image)[0][index]
print(model.predict(test_image))
# build the label
label = "{}: {:.2f}%".format(result, prob[0] * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (150, 300),  cv2.FONT_HERSHEY_SIMPLEX,
	1.0, (255, 0, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)