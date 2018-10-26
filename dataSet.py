import cv2
import sqlite3

#print (cv2.__version__)
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#insert/update data to sqlite
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("C:/DevPrograms/sqlite/sqlite-tools-win32-x86-3250200/FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name='"+str(Name)+"' WHERE ID='"+str(Id) +"'"
    else:
        cmd="INSERT INTO People(Id,Name) Values('"+str(Id)+"','"+str(Name)+"')"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
id=input('enter your id')
name=input('enter your name')
insertOrUpdate(id,name)
sampleNum=0
while(True):
    #camera read
    #ret, img = cam.read() 
    img = cv2.imread('D:/Training Python/Image/L.jpg',1)
    #gray = cv2.imread('D:/Training Python/Image/3.jpg', 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, 
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
