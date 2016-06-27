import os

im_dir = '/Users/philipppushnyakov/Downloads/roof_images_2/'
train_dir = '/Users/philipppushnyakov/DSG16/data/train/'
val_dir = '/Users/philipppushnyakov/DSG16/data/validation/'
im = {}
for i in range(4):
    im[i+1] = []

with open('/Users/philipppushnyakov/Downloads/id_train.csv') as fin:
    for line in fin:
        if not line.startswith('I'):
            parts = line.strip().split(',')
            im[int(parts[1])].append(parts[0])

for k,v in im.items():
    print len(im[k])


for i in range(4):
    for j in range(len(im[i+1])):
        if j<200:
            os.rename(im_dir + im[i+1][j] + '.jpg', val_dir + str(i+1) + '/' + im[i+1][j] + '.jpg')
        else:
            os.rename(im_dir + im[i+1][j] + '.jpg', train_dir + str(i+1) + '/' + im[i+1][j] + '.jpg')