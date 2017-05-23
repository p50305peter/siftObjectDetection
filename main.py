import cv2
import numpy as np
orb = cv2.xfeatures2d.SIFT_create()
dataset =[]
origin = []
sets=[]
name=[]
colorArray = [(255,0,250),(255, 220, 0),(0,0,255),(0,255,0),(0, 211, 255)]
for i in range(5):
    temp = cv2.imread("new_target"+str(i)+".png",1)
    name.append("new_target"+str(i))
    gray = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    dataset.append(gray)
    img = cv2.resize(temp, (120, 108));
    cv2.rectangle(img,(0,0),(120,108),colorArray[i],thickness=3)
    origin.append(temp)
    sets.append(img)
Target = np.vstack(sets)
TargetHeight = Target.shape[1];
TargetWidth = Target.shape[0];
videoCapture=cv2.VideoCapture("new_test1.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width = videoCapture.get(3)  # float
height = videoCapture.get(4) # float
out = cv2.VideoWriter('output2.avi', fourcc, 30, (int(1080),int(540)))
while (videoCapture.isOpened()):
    _ret,frame = videoCapture.read()
    #Target.resize(Target.shape[0],frame.shape[1])
    #print(str(Target.shape[0])+" "+str(Target.shape[1]))
    #print(str(frame.shape[0])+" "+str(frame.shape[1]))
    if(_ret == False):
        break;
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    img3 = frame
    cv2.line(frame,(0,0),(0,540),(255,0,0),thickness=5)
    for i in range(5):
        kpOrigin, des1 = orb.detectAndCompute(dataset[i], None)

        kpTarget, des2 = orb.detectAndCompute(gray, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good=[]
        for m,n in matches:
            #print(m.distance)
            if m.distance < 0.79 * n.distance:
                good.append(m)
        #matches = sorted(matches, key=lambda x: x.distance)
        print(len(good))
        temp=0
        flag=False
        if len(good) > 85:
            src_pts = np.float32([kpOrigin[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpTarget[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = origin[i].shape[:2]
            #print("pts")
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            #print(pts)
            dst = cv2.perspectiveTransform(pts, M)
            #print("dst")
            #print(dst)
            img3 = cv2.polylines(img3, [np.int32(dst)], True, colorArray[i], 3, cv2.LINE_AA)
            print(name[i]+" founded !")
            cv2.putText(img3,name[i]+" founded !",(int(dst[1][0][0]-20),int(dst[1][0][1]+30)),cv2.FONT_HERSHEY_SIMPLEX,1,colorArray[i],3)
            #cv2.putText(img3, name[i] + " fuck !" + " i ", (int(pts[1][0][0]), int(pts[1][0][1] + 30)),
                       # cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            #print(pts[3][0])
            temp= (int(dst[1][0][0]),int(dst[1][0][1]))
            cv2.line(img3, (0, 108*i+54), tuple(temp), colorArray[i], thickness=2)
            flag=True
           # print("ggggg")
            #print(dst)
        #cv2.findHomography(np.array(kpOrigin),np.array(kpTarget))
        view = np.hstack((Target, img3))
        if(flag==True):
            print((60,108*i+54))
            print((120+int(temp[0]),int(temp[1])))
            #cv2.line(view,(108*i+54,60),(120+int(temp[0]),int(temp[1])),colorArray[i],thickness=2)
    #view[:Target.shape[0],frame.shape[0]:frame.shape[1]] = img3
    out.write(view)
    cv2.imshow("test",view)
    print((view.shape[0],view.shape[1]))
    #cv2.waitKey(0)

out.release()
videoCapture.release()
cv2.destroyAllWindows()