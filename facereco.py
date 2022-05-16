import cv2
import numpy as np
import face_recognition
import os.path
import datetime
import time 

path = "face-recognition"
img  = []
className = []
mylist = os.listdir(path)


for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    img.append(curimg)
    className.append(os.path.splitext(cl)[0])



def findEncodings(img):
    encodeList = []
    for image in img:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

encodeListknown = findEncodings(img)
print("encoding complete")


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'n{name}, {time}, {date}')


cap = cv2.VideoCapture(0)

a = str(input("For new registration type yes/no : "))


if a == "yes":
    
    nameForNewReg = str(input("Enter your name : "))
    time.sleep(3)
    print("please be ready !! ")
    time.sleep(1)
    print("1")
    time.sleep(2)
    print("2")
    time.sleep(2)
    print("3")

    result,img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.250,0.25)
    faceCurFrmae = face_recognition.face_locations(imgs) 


    cv2.imshow("webcam",img)

    cv2.imwrite(os.path.join(path , f'{nameForNewReg}.jpg'), img)


elif a =="no":

    while True:
        success, img = cap.read()
        imgs = cv2.resize(img,(0,0),None,0.250,0.25)

        faceCurFrmae = face_recognition.face_locations(imgs) 
        encodecurframe = face_recognition.face_encodings(imgs,faceCurFrmae)

        for encodeface,faceloc in zip(encodecurframe,faceCurFrmae):
            matches = face_recognition.compare_faces(encodeListknown,encodeface)
            facedir = face_recognition.face_distance(encodeListknown,encodeface)

            matchindex = np.argmin(facedir)  

            name = className[matchindex].upper()
            print(name)  
            
            for i in matches:
                if i == True:
                    print("face recognised")
                    break
                                            
                else :
                    print("face not  recognised") 
                    break   

            if matches[matchindex]:
                
                y1, x2,y2,x1  = faceloc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)

            
            else :

                y1, x2,y2,x1  = faceloc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,"face not recognised",(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

                print ("please register your face for attendense")

            

        cv2.imshow('webcam',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




        



