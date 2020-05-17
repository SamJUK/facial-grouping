import cv2
import math
import argparse
import os
import json

# Input Aargs
parser = argparse.ArgumentParser()
parser.add_argument('--images')
args = parser.parse_args()

# Defintions
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']
outputData = {
	"overview": {},
	"lists": {},
	"files": []
}

# Setup Convoluted Neural Networks
ageNet = cv2.dnn.readNet("networks/age/age_net.caffemodel", "networks/age/age_deploy.prototxt")
genderNet = cv2.dnn.readNet("networks/gender/gender_net.caffemodel", "networks/gender/gender_deploy.prototxt")


# Process Images to array
print('*******************************')
print(f'Scanning source: {args.images}')
print('*******************************')

if os.path.isdir(args.images):
	images = [os.path.join(args.images, f) for f in os.listdir(args.images)]
else:
	images = [args.images]


# Logic
res = {
	'gender': {},
	'age': {}
}
for image_name in images:
	# Load current image
	image = cv2.imread(image_name)
	blob = cv2.dnn.blobFromImage(image, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

	# Calculate Gender
	genderNet.setInput(blob)
	gender = genderList[genderNet.forward()[0].argmax()]

	# Calculate Age
	ageNet.setInput(blob)
	age = ageList[ageNet.forward()[0].argmax()]

	# Increment Data store
	res['gender'][gender] = res['gender'].setdefault(gender, 0) + 1
	res['age'][age] = res['age'].setdefault(age, 0) + 1

	# Print output to console
	print(f'{image_name} :: {gender}, {age}')
	outputData['files'].append({
		'image_name': image_name,
		'gender': gender,
		'age': age
		})



# Sort results for output
total_age = sum(res['age'].values())
total_gender = sum(res['gender'].values())

gender_list = sorted(list(map(lambda gender: {
		'label': gender,
		'count': res['gender'][gender],
		'percent': round(res['gender'][gender] / total_gender * 100)
		}, res['gender'])), key=lambda k: k['count'], reverse=True)

age_list = sorted(list(map(lambda age: {
		'label': age,
		'count': res['age'][age],
		'percent': round(res['age'][age] / total_age * 100)
		}, res['age'])), key=lambda k: k['count'], reverse=True)

outputData['lists']['age'] = age_list
outputData['lists']['gender'] = gender_list



# Display to user
print('*******************************')
gender = gender_list[0]
print('- Gender: {} ({}%)'.format(gender['label'], gender['percent']))
outputData['overview']['gender'] = gender

age = age_list[0]
print('- Age: {} ({}%)'.format(age['label'], age['percent']))
outputData['overview']['age'] = age

print('*******************************')
print('# Genders')

for i in gender_list:
	print(' - {}: {} ({}%)'.format(i['label'], i['count'], i['percent']))

print('# Ages')

for i in age_list:
	print(' - {}: {} ({}%)'.format(i['label'], i['count'], i['percent']))


# Write file data
file_name = os.path.dirname(images[0]) + '/age_gender_output.json'
with open(file_name, 'w', encoding='utf-8') as file:
    json.dump(outputData, file, ensure_ascii=False, indent=4)









	