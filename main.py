import cv2
import numpy as np
import dlib
from deepface import DeepFace
from math import hypot
from time import sleep

Y_OFFSET = -50
POSSIBLE_EMOTIONS = ['happy', 'sad', 'neutral']

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("databases/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

def classify_emotion(frame):
    cv2.imwrite("imgs/face_input.png", frame)
    try:
        obj = DeepFace.analyze(img_path="imgs/face_input.png", actions=['emotion'])
        prob_happy = round(obj['emotion']['happy'])
        prob_sad = round(obj['emotion']['sad'])
        prob_neutral = round(100 - (prob_happy + prob_sad))
        if obj['dominant_emotion'] in POSSIBLE_EMOTIONS:
            return obj['dominant_emotion'], [prob_happy, prob_sad, prob_neutral]
        else:
            return 'neutral', [prob_happy, prob_sad, prob_neutral]
    except:
        return None, [0,0,0]


def add_emoji(frame, nose_mask, emoji_image):
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        # Nose coordinate
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 1.5)
        nose_height = int(nose_width)
        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2),
                    int(center_nose[1] - nose_height / 2) + Y_OFFSET)
        # Adding the new nose
        nose_emoji = cv2.resize(emoji_image, (nose_width, nose_height))
        nose_emoji_gray = cv2.cvtColor(nose_emoji, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_emoji_gray, 0, 30, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                          top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_emoji)
        frame[top_left[1]: top_left[1] + nose_height,
              top_left[0]: top_left[0] + nose_width] = final_nose


happy_emoji = cv2.imread("emojis/happy_face.png")
sad_emoji = cv2.imread("emojis/sad_face.png")
neutral_emoji = cv2.imread("emojis/neutral_face.png")
emojis = {'happy': happy_emoji, 'sad': sad_emoji, 'neutral': neutral_emoji}

def webcam_func():
    cap = cv2.VideoCapture(0)

    _, frame = cap.read()
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)

    while True:
        _, frame = cap.read()
        emotion, probs = classify_emotion(frame)
        p_happy, p_sad, p_neutral = probs
        if emotion is not None:
            emoji = emojis[emotion]
            add_emoji(frame, nose_mask, emoji)
            frame = cv2.putText(frame, f"You are {p_happy}% happy", (0, 50), font, fontScale, color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, f"You are {p_sad}% sad", (0, 125), font, fontScale, color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, f"You are {p_neutral}% neutral", (0, 200), font, fontScale, color, thickness, cv2.LINE_AA, False)
        #sleep(1)
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            break


def video_func():
    file_name = input("Enter video file name: ")
    cap = cv2.VideoCapture(file_name)
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (cols//3, rows//3))
        emotion, probs = classify_emotion(frame)
        p_happy, p_sad, p_neutral = probs
        if emotion is not None:
            emoji = emojis[emotion]
            add_emoji(frame, nose_mask, emoji)
            frame = cv2.putText(frame, f"You are {p_happy}% happy", (0, 50), font, fontScale, color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, f"You are {p_sad}% sad", (0, 125), font, fontScale, color, thickness, cv2.LINE_AA, False)
            frame = cv2.putText(frame, f"You are {p_neutral}% neutral", (0, 200), font, fontScale, color, thickness, cv2.LINE_AA, False)
        #sleep(1)
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

def image_func():
    file_name = input("Enter image file name: ")
    frame = cv2.imread(f'imgs/{file_name}')
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)

    frame = cv2.resize(frame, (cols//3, rows//3))
    emotion, probs = classify_emotion(frame)
    p_happy, p_sad, p_neutral = probs
    if emotion is not None:
        emoji = emojis[emotion]
        add_emoji(frame, nose_mask, emoji)
        frame = cv2.putText(frame, f"You are {p_happy}% happy", (0, 50), font, fontScale, color, thickness, cv2.LINE_AA, False)
        frame = cv2.putText(frame, f"You are {p_sad}% sad", (0, 125), font, fontScale, color, thickness, cv2.LINE_AA, False)
        frame = cv2.putText(frame, f"You are {p_neutral}% neutral", (0, 200), font, fontScale, color, thickness, cv2.LINE_AA, False)

    cv2.imshow("Frame", frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_options() -> None:
    print("-" * 25)
    print("1) Use Image File")
    print("2) Use Video File")
    print("3) Use Webcam")
    print("4) Quit")

def get_option() -> int:
    print_options()
    option = int(input("Enter your option: "))
    print("-" * 25)
    return option

def welcome_message():
    print("Welcome to the AI Face Filter!")

def bye_message():
    print("Good bye, thanks for using the program!")

def main():
    welcome_message()
    option = get_option()
    while option != 4:
        if option == 1:
            image_func()
        elif option == 2:
            video_func()
        elif option == 3:
            webcam_func()

        option = get_option()
    bye_message()

if __name__ == '__main__':
    main()