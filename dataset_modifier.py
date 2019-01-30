import random
import os
import cv2

MAIN_PATH = '/home/pranav/PycharmProjects/Fingerprint live detection/liveDet_data/Digital_Persona'
NEW_PATH = '/home/pranav/PycharmProjects/Fingerprint live detection/liveDet_data/Digital_Persona_shuffled/'

if(os.path.isdir(NEW_PATH) == False):
        os.mkdir(NEW_PATH)

dirs = os.listdir(MAIN_PATH)
fake_files_original = []

live_files_orginal = []
path_original = dirs[0]
files =  os.listdir(os.path.join(MAIN_PATH,path_original))
for file in files:
    print(os.path.join(path_original,file))
    fake_files_original.append(os.path.join(path_original,file))



path_original = dirs[1]
files =  os.listdir(os.path.join(MAIN_PATH,path_original))

for file in files:
    print(os.path.join(path_original,file))
    live_files_orginal.append(os.path.join(path_original,file))


random.shuffle(fake_files_original)
random.shuffle(live_files_orginal)

train_fake_files_original = fake_files_original[0:int(0.8*len(fake_files_original))]
val_fake_files_original = fake_files_original[int(0.8*len(fake_files_original)):int(0.9*len(fake_files_original))]
test_fake_files_original = fake_files_original[int(0.9*len(fake_files_original)):len(fake_files_original)]


train_live_files_original = live_files_orginal[0:int(0.8*len(live_files_orginal))]
val_live_files_original = live_files_orginal[int(0.8*len(live_files_orginal)):int(0.9*len(live_files_orginal))]
test_live_files_original = live_files_orginal[int(0.9*len(live_files_orginal)):len(live_files_orginal)]

print(len(train_fake_files_original))
print(len(val_fake_files_original))
print(len(test_fake_files_original))

dirs = ['train','val','test'] ## train test


for directory in dirs:
    print(directory)
    #print(os.path.join(NEW_PATH,directory))
    os.mkdir(os.path.join(NEW_PATH,directory))

dirs = ['train','val','test'] ## train test
for directory in dirs:
    if(os.path.isdir(os.path.join(os.path.join(NEW_PATH,directory),'Fake')) == False):
        os.mkdir(os.path.join(os.path.join(NEW_PATH,directory),'Fake'))
    if(os.path.isdir(os.path.join(os.path.join(NEW_PATH,directory),'Live')) == False):
        os.mkdir(os.path.join(os.path.join(NEW_PATH, directory), 'Live'))


#os.mkdir(os.path.join(NEW_PATH, 'train'))


#class_dirs = os.listdir(os.path.join(NEW_PATH,directory))
path = os.path.join(NEW_PATH,dirs[0])


for file in train_fake_files_original:
    img = cv2.imread(os.path.join(MAIN_PATH,file))
    print(os.path.join(path,file))
    cv2.imwrite(os.path.join(path,file), img)

path = os.path.join(NEW_PATH,dirs[0])

for file in train_live_files_original:
    img = cv2.imread(os.path.join(MAIN_PATH,file))
    print(os.path.join(path,file))
    cv2.imwrite(os.path.join(path,file), img)

path = os.path.join(NEW_PATH,dirs[1])


for file in val_fake_files_original:
    img = cv2.imread(os.path.join(MAIN_PATH,file))
    print(os.path.join(path,file))
    cv2.imwrite(os.path.join(path,file), img)

path = os.path.join(NEW_PATH,dirs[1])

for file in val_live_files_original:
    img = cv2.imread(os.path.join(MAIN_PATH,file))
    print(os.path.join(path,file))
    cv2.imwrite(os.path.join(path,file), img)

path = os.path.join(NEW_PATH,dirs[2])


for file in test_fake_files_original:
    img = cv2.imread(os.path.join(MAIN_PATH,file))
    print(os.path.join(path,file))
    cv2.imwrite(os.path.join(path,file), img)

path = os.path.join(NEW_PATH,dirs[2])

for file in test_live_files_original:
    img = cv2.imread(os.path.join(MAIN_PATH,file))
    print(os.path.join(path,file))
    cv2.imwrite(os.path.join(path,file), img)

