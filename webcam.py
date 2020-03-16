import cv2
import pickle

with open('face_recog_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
    pipeline.steps[1][1].load_model('models/keras/facenet_keras.h5')
    
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    # print(type(frame), frame.shape)
    try:
        print(pipeline.predict([frame]))
    except:
        pass
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")