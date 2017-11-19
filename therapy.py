import cv2
from keras.models import load_model
import numpy as np
import glob 

# parameters for loading data and images
emotion_model_path = '..\\trained_models\\fer2013_big_XCEPTION.54-0.66.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
positive = ["happy","neutral"]
negative = ["angry",'disgust','fear','sad']
songs_at = "..\\music1\\"
therapy_sounds = {"anger":[1,8,10,9],"surprise":[2,5,7,14],"disgust":[8,10,3,4,13],"fear":[10,9,12,13],"sad":[3,9,12,13],"happy":[1],'neutral':[1]}
leave_song = False
mood_other_way = False

# loading models
#IMPORTANT HARRCASCADE addresss
cascade_file_src = "..\\trained_models\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_file_src)
emotion_clf = load_model(emotion_model_path, compile=False)


#Trial video
cap = cv2.VideoCapture("trial.mp4")
#cap = cv2.VideoCapture(0)
while(True):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    all_ppl = len(faces)
    if all_ppl == 0:
        continue

    for_emotion = np.zeros((all_ppl,64,64,1))
    #assumed for multiple faces!
    for index, face in enumerate(faces):
        (x, y, w, h) = face
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ith_face = gray[y:y+h,x:x+w] 
        ith_face = cv2.resize(ith_face,(64,64))
        ith_face = ith_face.astype('float32')
        ith_face = ith_face / 255.0
        ith_face = ith_face - 0.5
        ith_face = ith_face * 2.0
        for_emotion[index] = ith_face.reshape((64,64,1))
    #searching for most prominent emotion
    emotion_prediction = emotion_clf.predict(for_emotion)
    og_emotion_prob = np.max(emotion_prediction)
    which_emotion = np.argmax(emotion_prediction)
    og_emotion_text = emotion_labels[which_emotion]
    print(og_emotion_text)
    #Prescription
    
    if og_emotion_text in negative:
        while(True):
            song_prescription = therapy_sounds[og_emotion_text]
            for song in song_prescription:
                all_parts = glob.glob(songs_at + str(song) + "_*") 
                for part in all_parts:
                    playsound.playsound(part)
                    ret, image = cap.read()
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                    for_emotion = np.zeros((all_ppl,64,64,1))
                    #assumed for multiple faces!
                    for index, face in enumerate(faces):
                        (x, y, w, h) = face
                        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        ith_face = gray[y:y+h,x:x+w] 
                        ith_face = cv2.resize(ith_face,(64,64))
                        ith_face = ith_face.astype('float32')
                        ith_face = ith_face / 255.0
                        ith_face = ith_face - 0.5
                        ith_face = ith_face * 2.0
                        for_emotion[index] = ith_face.reshape((64,64,1))
                    #searching for most prominent emotion
                    emotion_prediction = emotion_clf.predict(for_emotion)
                    new_emotion_prob = np.max(emotion_prediction)
                    which_emotion = np.argmax(emotion_prediction)
                    new_emotion_text = emotion_labels[which_emotion]
                    print(new_emotion_text)
                    if new_emotion_text in negative:
                        if new_emotion_text == og_emotion_text:
                            if new_emotion_prob - og_emotion_prob <= 0:
                                continue
                            else:
                                leave_song = True
                        else:
                            mood_other_way = True
                    if leave_song:
                        leave_song = False
                        break
                if mood_other_way:
                    mood_other_way = False
                    song_prescription = therapy_sounds[new_emotion_text]
                    og_emotion_text = new_emotion_text
                    og_emotion_prob = new_emotion_prob
                    break
                og_emotion_text = new_emotion_text
                og_emotion_prob = new_emotion_prob
                    
    
		