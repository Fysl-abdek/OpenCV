# import required packge
import cv2

def faceBox(frame, net):
   height = frame.shape[0]
   width = frame.shape[1]

   blob = cv2.dnn.blobFromImage(cv2.resize(
       frame, (227, 227)), 1.0, (227, 227), [104.0, 177.0, 123.0])

   # pass the blob through the network and obtain the detections and predictions
   net.setInput(blob)
   detections = net.forward()
   bboxs = []

   # loop over detectons
   for i in range(detections.shape[2]):

      confidence = detections[0, 0, i, 2]

      # compute the (x, y)-coordinates of the bounding box for the object
      if confidence > 0.5:
         startX = int(detections[0, 0, i, 3] * width)
         startY = int(detections[0, 0, i, 4] * height)
         endX = int(detections[0, 0, i, 5] * width)
         endY = int(detections[0, 0, i, 6] * height)

         # # draw the bounding box of the face along with the associated probability
         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
         bboxs.append([startX, startY, endX, endY])
   return (frame, bboxs)


# define some prameter
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20


video = cv2.VideoCapture(0)

# face detect pre_train model
model = "res10_300x300_ssd_iter_140000.caffemodel"
prototxt = "deploy.prototxt.txt"

# age classicifcation model
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

# read age net
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# gender model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# read gender net
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# read the face detect model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

while True:
   frame = video.read()[1]
   face = frame.copy()
   frame, bboxs = faceBox(frame, net)

   for bbox in bboxs:
      face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                   max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
      blob = cv2.dnn.blobFromImage(
          face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

      # age prediction
      ageNet.setInput(blob)
      agePred = ageNet.forward()
      age = ageList[agePred[0].argmax()]

      # gender prediction
      genderNet.setInput(blob)
      genderPred = genderNet.forward()
      gender = genderList[genderPred[0].argmax()]

      # put age and gender in the top of bounding box
      text = '{},{}'.format(gender, age)

      # cv2.rectangle(frame, (bbox[0], bbox[1]-30),(bbox[2], bbox[1]), (0, 255, 0), -1)
      cv2.putText(frame, text, (bbox[0], bbox[1]-10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

   cv2.imshow("face_detected", frame)
   k = cv2.waitKey(1)
   if k == ord("q"):
      break
video.release()
cv2.destroyAllWindows()
