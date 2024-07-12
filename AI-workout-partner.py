import cv2 
# from markdown import Markdown
import numpy as np
import mediapipe as mp
import pyttsx3
from functions import calculate_angle, progress_bar, counter_box, alert, render_angle_and_line
import threading
import pathlib
import textwrap

# import google.generativeai as genai
text_speech = pyttsx3.init()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def to_markdown(text):
  text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# GOOGLE_API_KEY="AIzaSyCSc7mz3nIPxI7eLCfTOlEiUyRnXoGic7Q"

# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-pro')

choice=0
while choice!=5:
    choice = int(input("Enter the exercise (Curls: 0, Squats: 1, Pushups: 2, Shoulder Press: 3,Ask questions: 4, Exit: 5): "))

# captures the video
    cap = cv2.VideoCapture(0)

# counter and stage variables
    count = 0
    stage = None

    if choice == 0:     #curls
        first = 11      #left_shoulder
        second = 13     #left_elbow
        third = 15      #left_wrist
        down_threshold = 160
        up_threshold = 30
        stage_type = 0
        exercise = 'Curls'
        fourth = 23     #left_hip
        fifth = 11      #left_shoulder
        sixth = 13      #left_elboy
        seventh=11      #left_shoulder
        eighth=23       #
        ninth=25
    elif choice == 1:   #squats
        first = 23
        second = 25
        third = 27
        down_threshold = 160
        up_threshold = 80
        stage_type = 1
        exercise = 'Squats'
    elif choice == 2:   #pushups
        first = 11
        second = 13
        third = 15
        down_threshold = 160
        up_threshold = 90
        stage_type = 1
        exercise = 'Push-ups'
        fourth = 27
        fifth = 23
        sixth = 11
    elif choice == 3:   #shoulder press
        first = 11
        second = 13
        third = 15
        down_threshold = 160
        up_threshold = 80
        stage_type = 1
        exercise = 'Shoulder press'
        fourth = 12
        fifth = 14
        sixth = 16
    elif choice == 4:     #chatbot
        x='a'
        while(x != 'q'):
            x = input("Ask (press q to exit): ")
            # response = model.generate_content(x)
            # print(response.text)
        exit(0)
    elif choice == 5:     #exit
        exit()
    else:
        print('invalid')
        exit()
    # setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()  # takes in the frames

            # recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # make detection
            results = pose.process(image)

            # recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # extract landmarks
            try:
                if results.pose_landmarks is not None:
                    landmarks = results.pose_landmarks.landmark
                    en=0
                    # get coordinates
                    a = [landmarks[first].x, landmarks[first].y]
                    b = [landmarks[second].x, landmarks[second].y]
                    c = [landmarks[third].x, landmarks[third].y]

                    # calculate angle
                    angle1 = calculate_angle(a, b, c)
                    angle1 = int(round(angle1, 3))

                    # visualise
                    cv2.putText(image, str(angle1),
                                tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    if choice != 1:
                        # get coordinates for form correction
                        d = [landmarks[fourth].x, landmarks[fourth].y]
                        e = [landmarks[fifth].x, landmarks[fifth].y]
                        f = [landmarks[sixth].x, landmarks[sixth].y]
                        if choice==0:
                           g=[landmarks[seventh].x,landmarks[seventh].y]
                           h=[landmarks[eighth].x,landmarks[eighth].y]
                           i=[landmarks[ninth].x,landmarks[ninth].y]
                           
                           angle3 = calculate_angle(g,h,i)
                           angle3 = int(round(angle3))
                           cv2.putText(image, str(angle3),
                                    tuple(np.multiply(h, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                           if angle3>180 or angle3<172:
                               en=1


                        # calculate angle
                        angle2 = calculate_angle(d, e, f)
                        angle2 = int(round(angle2))

                        # visualise
                        cv2.putText(image, str(angle2),
                                    tuple(np.multiply(e, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                    else:
                        angle2 = render_angle_and_line(image, landmarks)
                    if choice==0 and en==1:
                        alert(angle1,angle3,choice,image,count,en)
                    else:
                        alert(angle1, angle2, choice, image, count,en)


                    # counter
                    if angle1 > down_threshold:
                        stage = 'down'
                    if angle1 < up_threshold and stage == 'down':
                        stage = ' up'
                        count += 1

                    # Calculate the progress of the loading bar based on the angle
                    progress = 1 - ((angle1 - up_threshold) / (down_threshold - up_threshold))  # Normalize the angle between 0 and 1
                    progress_bar(progress, image)  # Loading bar

            except Exception as e:
                print("q")

            counter_box(image, count, stage, stage_type, exercise)  # Renders counter box

            # render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

            cv2.imshow('Mediapipe Feed', image)  # shows the video
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                # to exit press 'q'
                break