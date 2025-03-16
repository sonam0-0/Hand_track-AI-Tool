import cv2
import time
# import HandTrackingModule as htm

cap = cv2.VideoCapture(0)  # Change 0 to 1 if using an external camera
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
pTime=0
# detector=htm.handDetector()
while True:
    success, img = cap.read()
    # img=detector.findHands(img)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}',(40,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
    if not success:
        print("Error: Failed to capture image")
        break
    cv2.imshow("Img", img)
    # Exit the loop if the user presses the ESC key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break
cap.release()
cv2.destroyAllWindows()

