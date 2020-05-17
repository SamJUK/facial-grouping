import face_recognition
import cv2
import numpy as np
import uuid
import os
import re
import argparse
import statistics

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
parser.add_argument('--tolerance', nargs='?', const=1, type=float, default=.6)
parser.add_argument('--blur', nargs='?', const=1, type=int, default=0)
parser.add_argument('--framesize', nargs='?', const=1, type=int, default=1)
args=parser.parse_args()

# Open video file
video = args.video
video_capture = cv2.VideoCapture(video)

video_capture_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT));
print(f'Total Frames: {video_capture_frame_count}')

# sort inital folders
base_folder = 'faces/{}'.format(re.sub('[^0-9a-zA-Z]+', '_', video))
makeFolder(base_folder)

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

    # Resize frame of video can cause faster face recognition processing
    frame = cv2.resize(frame, (0, 0), fx=args.framesize, fy=args.framesize)

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
                matches = face_recognition.compare_faces(person['faces'], face, args.tolerance)
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
                    data['person']['faces'].append(face)
                    matched = True
            
            if not matched:
                tmp = {
                    'name': f'E{len(persons)+1}',
                    'faces': [face]
                }
                name = tmp['name']
                matchAccuracy = 1
                persons.append(tmp)

            print(f' - Found {name} ({matchAccuracy})')
            names.append(name)
            matchPs.append(matchAccuracy)

        # Display the resulting image
        output_frame = frame.copy()
        if args.blur > 0:
            output_frame = cv2.blur(output_frame, (args.blur, args.blur), cv2.BORDER_DEFAULT)

        for (top, right, bottom, left), name, matchP in zip(face_locations, names, matchPs):
            # Write to disk
            face_frame = frame.copy()
            person_folder = '{}/{}'.format(base_folder, name)
            makeFolder(person_folder)
            cv2.imwrite('{}/{}.jpg'.format(person_folder, str(uuid.uuid4())), face_frame[top:bottom, left:right])

            # Draw a box around the face
            cv2.rectangle(output_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(output_frame, '{} ({})'.format(name, round(matchP, 2)), (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),  1, cv2.LINE_AA)

        cv2.imshow('Video Facial Extraction Classification', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


averageFaces = statistics.mean(map(lambda p: len(p['faces']), persons))

print(f'Checking for potential false positive Threshold ({averageFaces})')
for person in persons:
    faceThreshold = averageFaces * .25
    faceCount = len(person['faces'])

    if len(person['faces']) < faceThreshold:
        person_name = person['name']
        print(f' - Potential false: {person_name} faces ({faceCount}/{faceThreshold})')

        for face in person['faces']:
            matchData = []
            for possiblePerson in persons:
                if possiblePerson == person:
                    continue

                matches = face_recognition.compare_faces(possiblePerson['faces'], face, args.tolerance)
                matchAccuracy = (sum(matches) / len(matches))
                matchData.append({
                    'person': possiblePerson,
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
                    persons_name = data['person']['name']
                    print(f'Face better matches {persons_name}')


video_capture.release()
cv2.destroyAllWindows()

print('Finished')
