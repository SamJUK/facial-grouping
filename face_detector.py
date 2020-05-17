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


def checkFaceMatchesPerson(face, skips = []):
    result = []
    for person in persons:
        if person in skips:
            continue

        matches = face_recognition.compare_faces(person['faces'], face, args.tolerance)
        matchAccuracy = (sum(matches) / len(matches))
        result.append({
            'person': person,
            'matchAccuracy': matchAccuracy
            })

    return result

def getClosestMatch(matches):
    bestMatch = {'matchAccuracy': -1}

    for match in matches:
        if match['matchAccuracy'] > bestMatch['matchAccuracy']:
            bestMatch = match

    return bestMatch


def determinePerson(skips = []):
    matchData = checkFaceMatchesPerson(face, skips)

    matched = False
    if len(matchData):
        data = getClosestMatch(matchData)

        matchAccuracy = data['matchAccuracy']
        if matchAccuracy > args.matchaccuracy:
            
            return {
                'person': data['person'],
                'accuracy': matchAccuracy
            }

    return False

def createNewPerson(face):
    newPerson = {
        'name': f'E{len(persons)+1}',
        'faces': [face]
    }
    
    persons.append(newPerson)

    return newPerson

parser = argparse.ArgumentParser()
parser.add_argument('--video')
parser.add_argument('--frames', nargs='?', const=1, type=float, default=.01)
parser.add_argument('--matchaccuracy', nargs='?', const=1, type=float, default=.15)
parser.add_argument('--tolerance', nargs='?', const=1, type=float, default=.6)
parser.add_argument('--blur', nargs='?', const=1, type=int, default=0)
parser.add_argument('--framesize', nargs='?', const=1, type=int, default=1)
args = parser.parse_args()


# Open video file
video = args.video
video_capture = cv2.VideoCapture(video)
video_capture_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT));
print(f'Total Frames: {video_capture_frame_count}')

# Create a folder for this video
alphanumeric_video_name = re.sub('[^0-9a-zA-Z]+', '_', video)
base_folder = f'faces/{alphanumeric_video_name}'
makeFolder(base_folder)

# Base definition for variables
face_encs = []
face_names = []
persons = []
saved_frames = []
frame_count = 0

# Process Video
while video_capture.isOpened():

    # Grab a single frame of video
    retval, frame = video_capture.read()

    # Break the loop after process whole video
    if not retval:
        break

    # Resize frame of video. Smaller frame size can improve face recognition processing speed
    frame = cv2.resize(frame, (0, 0), fx=args.framesize, fy=args.framesize)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    frame_count += 1

    # Skip Frames to help speed up processing
    if frame_count % round(video_capture_frame_count * args.frames) == 0:

        # Pull facial Data
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        face_accuracy = []
        for face in face_encodings:
            person = determinePerson()
            if person != False:
                name = person['person']['name']
                accuracy = person['accuracy']
                person['person']['faces'].append(face)
            else:
                name = createNewPerson(face)['name']
                accuracy = 1

            print(f' - Found {name} ({accuracy})')
            face_names.append(name)
            face_accuracy.append(accuracy)

        # Display the resulting image
        output_frame = frame.copy()
        if args.blur > 0:
            output_frame = cv2.blur(output_frame, (args.blur, args.blur), cv2.BORDER_DEFAULT)

        # Save faces, and highlight on output
        for (top, right, bottom, left), name, matchP in zip(face_locations, face_names, face_accuracy):
            # Write to disk
            file_name = str(uuid.uuid4())
            person_folder = f'{base_folder}/{name}'
            makeFolder(person_folder)
            cv2.imwrite(f'{person_folder}/{file_name}.jpg', frame.copy()[top:bottom, left:right])

            # Draw a box around the face, and text displaying person and match certancy 
            cv2.rectangle(output_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(output_frame, '{} ({})'.format(name, round(matchP, 2)), (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),  1, cv2.LINE_AA)

        # Render image
        cv2.imshow('Video Facial Extraction Classification', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


averageFaces = statistics.mean(map(lambda p: len(p['faces']), persons))

print(f'Checking for potential false positive Threshold ({averageFaces})')
for person in persons:
    # If the person has 25% or less face extractions that the average, we assume it may be a false positive
    faceThreshold = averageFaces * .25
    faceCount = len(person['faces'])

    if faceCount < faceThreshold:
        person_name = person['name']
        print(f' - Potential false: {person_name} faces ({faceCount}/{faceThreshold})')

        for face in person['faces']:
            person = determinePerson(face, [person])

            if person != False:
                print('Face better matches ' + data['person']['name'])


# Cleanup
video_capture.release()
cv2.destroyAllWindows()