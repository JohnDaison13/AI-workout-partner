import cv2 
import numpy as np
import mediapipe as mp
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if angle>180.0:
        angle=360-angle
    return angle



def loading_bar(progress):
    if progress < 0:
        progress = 0
    if progress > 1:
        progress = 1

    # Draw loading bar
    loading_bar_width = 20
    loading_bar_height = 250
    loading_bar_x = 20
    loading_bar_y = 350
    loading_bar_end_y = loading_bar_y - int(loading_bar_height * progress)

    colour=(0,255,0) #default green
    if progress>0.2:
        colour = (0, 255, 255)  # Yellow
    if progress >= 0.5:
        colour = (0, 165, 255)  # Orange
    if progress >= 0.8:
        colour = (0, 0, 255)  # Red

    cv2.rectangle(image, (loading_bar_x, loading_bar_y),
                    (loading_bar_x + loading_bar_width, loading_bar_y - loading_bar_height),
                    colour, 2) 

    cv2.rectangle(image, (loading_bar_x, loading_bar_y),
                    (loading_bar_x + loading_bar_width, loading_bar_end_y),
                    colour, -1)

    # Display progress percentage
    cv2.putText(image, f"{int(progress * 100)}%",
                (loading_bar_x, loading_bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    

def counter_box(image,count,stage):
    #render counter box
    
    cv2.rectangle(image,(8,8),(175,90),(0,0,0),-1)

    cv2.putText(image,excercise,
                (16,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
    
    #render rep data
    cv2.putText(image,"REPS",
                (16,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,str(count),
                (18,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

    #render stage data
    if stage_type==1:
        if stage=='down':
            stage=' up'
        else:
            stage='down'
    
    cv2.putText(image,"STAGE",
                (75,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,stage,
                (65,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

#captures the video
cap=cv2.VideoCapture(0)

#counter and stage variables
count=0
stage=None

choice = int(input("Enter the excercise (Curls:0, Squats:1, Pushups:2, Shoulder Press:3): "))

if choice==0:
    first=11
    second=13
    third=15
    down_threshold=160
    up_threshold=30
    stage_type=0
    excercise='Curls'

elif choice==1:
    first=23
    second=25
    third=27
    down_threshold=160
    up_threshold=80
    stage_type=1
    excercise='Squats'


elif choice==2:
    first=11
    second=13
    third=15
    down_threshold=160
    up_threshold=90
    stage_type=1
    excercise='Push-ups'

elif choice==3:
    first=11
    second=13
    third=15
    down_threshold=160
    up_threshold=50
    stage_type=1
    excercise='Shoulder press'

else:
    print("Invalid")
    exit()

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:

    while cap.isOpened():
        ret,frame=cap.read()        #takes in the frames

        #recolor image to RGB
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable= False

        #make detection
        results=pose.process(image)

        #recolor back to BGR
        image.flags.writeable= True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #extract landmarks
        try:
            landmarks=results.pose_landmarks.landmark

            #get coordinates
            a=[landmarks[first].x,landmarks[first].y]
            b=[landmarks[second].x,landmarks[second].y]
            c=[landmarks[third].x,landmarks[third].y]

            #calculate angle
            angle=calculate_angle(a,b,c)
            angle = int(round(angle, 3))
            
            #visualise
            cv2.putText(image,str(angle),
                        tuple(np.multiply(b,[640,480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                        )
            
            #counter
            if angle > down_threshold:
                stage='down'
            if angle < up_threshold and stage=='down':
                stage=' up'
                count+=1

              # Calculate the progress of the loading bar based on the angle
            progress = 1-((angle - up_threshold) / (down_threshold - up_threshold))  # Normalize the angle between 0 and 1
            
            loading_bar(progress)       #-------- RENDERS LOADING BAR 

        except:
            pass

        counter_box(image,count,stage)      #------ RENDERS COUNTER BOX


        #render detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                 )


        cv2.imshow('Mediapipe Feed',image)      #shows the video

        if cv2.waitKey(10) & 0xFF==ord('q'):    #to exit press 'q'
            break
    cap.release()
    cv2.destroyAllWindows()     #exits