#PSEUDOCODE
import videostream
import transform

tranformObject = transform.Transform()

# 1. Grab image from PiCam (or webcam if present)
stream = videostream.VideoStream(resolution=(1280,720),framerate=30).start()
img = stream.read()
if img == None:
    print('Camera not found')
    stream.stop()
    exit()

# 2. Transform the image using a build in function, make sure the image is ready
img = tranformObject.numberTransform(img)

# 3. Load the model

# 4. Make the model do a prediction
# 5. Display the prediction (on screen or in the console)