import json, os, cv2
from tqdm import tqdm

# ================================================
# dict 파일 생성
# ================================================

path = '/opt/ml/project/data/'

train = json.load(open(path + 'train.json', 'r'))
val = json.load(open(path + 'val.json', 'r'))
print('train, val')
print(len(train), len(val), '\n')

info = {}

def Update(info, src):
    for img in tqdm(src):
        img_id = int(img.split('_')[0])

        left_eye = src[img]['left_eye']
        right_eye = src[img]['right_eye']
        nose = src[img]['nose']
        mouth = src[img]['mouth']
        remainder = src[img]['face_fit']

        Conv = lambda x: int(round(x))
        left_eye = [*map(Conv, left_eye)]
        right_eye = [*map(Conv, right_eye)]
        nose = [*map(Conv, nose)]
        mouth = [*map(Conv, mouth)]
        remainder = [[*map(Conv, i)] for i in remainder]

        info[img_id] = {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'nose': nose,
            'mouth': mouth,
            'remainder': [remainder[0], remainder[2]]
        }

Update(info, train)
Update(info, val)
print('info : ')
print(len(info), '\n')



# ================================================
# image 다른 폴더에 이름 바꿔서 저장
# ================================================

img_to_idx = {}
for idx, img in enumerate(info):
    img_to_idx[img] = idx

def SaveImage(img_folder_path, save_folder_path):
    img_list = os.listdir(img_folder_path)
    for img_name in tqdm(img_list):
        img_id = int(img_name.split('_')[0])
        img = cv2.imread(img_folder_path + img_name)
        save_path = save_folder_path + str(img_to_idx[img_id]) + '.png'
        cv2.imwrite(save_path, img)

# SaveImage(path + '_image/', path + 'image/')
# SaveImage(path + '_sketch/', path + 'sketch/')
print()



# ================================================
# train.json, val.json 저장
# ================================================

d = {}
for i in tqdm(info):
    d[img_to_idx[i]] = info[i]
    d[img_to_idx[i]]['image_path'] = path + 'image/' + str(img_to_idx[i]) + '.png'
    d[img_to_idx[i]]['sketch_path'] = path + 'sketch/' + str(img_to_idx[i]) + '.png'

train_d = {}
val_d = {}
for i in range(len(d)):
    if i < 3600: train_d[i] = d[i]
    else: val_d[i] = d[i]

def SaveJson(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent = 4)

SaveJson(path + '_all.json', d)
SaveJson(path + '_train.json', train_d)
SaveJson(path + '_val.json', val_d)
