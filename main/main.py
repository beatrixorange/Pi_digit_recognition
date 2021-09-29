from tensorflow import keras
import matplotlib.pyplot as plt
import videostream
import transform

# Load the pre-trained model
model = keras.models.load_model('./model.h5')

# Prepare the transform object
tranformObject = transform.Transform()

# Grab image from PiCam (or webcam if present)
stream = videostream.VideoStream(resolution=(1280,720),framerate=30).start()
img = stream.read()
if img is None:
    print('Camera not found')
    stream.stop()
    exit()

# Transform the image using a build in function, make sure the image is ready
img = tranformObject.numberTransform(img)

# Make the model do a prediction
pred = model.predict(img)

#  Display the prediction (on screen or in the console)
print(pred.argmax())
print(pred)
plt.imshow(img)
plt.show()