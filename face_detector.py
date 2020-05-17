import face_recognition
import cv2
import numpy as np
import uuid
import os
import re
import argparse

def makeFolder(path): 
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


parser=argparse.ArgumentParser()
parser.add_argument('--video')
parser.add_argument('--frames', nargs='?', const=1, type=float, default=.01)
parser.add_argument('--matchaccuracy', nargs='?', const=1, type=float, default=.15)
args=parser.parse_args()

# Open video file
video = args.video
video_capture = cv2.VideoCapture(video)

video_capture_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT));
print('Total Frames: {}'.format(video_capture_frame_count))

# sort inital folders
base_folder = 'output/{}'.format(re.sub('[^0-9a-zA-Z]+', '_', video))
faces_folder = '{}/faces'.format(base_folder)
makeFolder(base_folder)
makeFolder(faces_folder)

face_encs = []
face_names = []

persons = []
saved_frames = []
frame_count = 0

if args.frames == -1:
    args.frames = video_capture_frame_count

while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Bail out when the video file ends
    if not ret:
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)


    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    frame_count += 1


    # Skip Frames to save time 
    if frame_count % round(video_capture_frame_count * args.frames) == 0:

        # Pull Data
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names = []
        matchPs = []
        for face in face_encodings:
            matchData = []
            for person in persons:
                matches = face_recognition.compare_faces(person['faces'], face)
                matchAccuracy = (sum(matches) / len(matches))
                matchData.append({
                    'person': person,
                    'matchAccuracy': matchAccuracy
                    })

            matched = False
            if len(matchData):
                data = {'matchAccuracy':-1}

                for d in matchData:
                    if d['matchAccuracy'] > data['matchAccuracy']:
                        data = d

                matchAccuracy = data['matchAccuracy']

                if matchAccuracy > args.matchaccuracy:
                    name = data['person']['name']
                    person['faces'].append(face)
                    matched = True
            
            if not matched:
                tmp = {
                    'name': 'E{}'.format(len(persons)+1),
                    'faces': [face]
                }
                name = tmp['name']
                matchAccuracy = 1
                persons.append(tmp)

            print(' - Found {} ({})'.format(name, matchAccuracy))
            names.append(name)
            matchPs.append(matchAccuracy)


        for (top, right, bottom, left), name, matchP in zip(face_locations, names, matchPs):
            # Write to disk
            of = frame.copy()
            fname = str(uuid.uuid4())
            person_folder = '{}/{}'.format(faces_folder, name)
            makeFolder(person_folder)
            cv2.imwrite('{}/{}.jpg'.format(person_folder,fname), of[top:bottom, left:right])

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, '{} ({})'.format(name, round(matchP, 2)), (left + 6, bottom - 6), font, .5, (255, 255, 255),  1, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video_capture.release()
cv2.destroyAllWindows()


