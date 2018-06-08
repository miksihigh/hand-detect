import numpy as np
from matplotlib import pyplot as pl
import pandas
from sklearn.linear_model import LinearRegression
from neural_network import FaceNet
from PIL import Image
import matplotlib.patches as patches
data = pandas.read_csv('data.csv', delimiter=' ')
pathes = data[['File']].as_matrix()[:, 0]
rects = data[['x', 'y', 'w', 'h']].as_matrix()
size = 48
imgX = []
rectY = []


for file, rect in zip(pathes, rects):
    img = Image.open(file)
    x = rect[0] / img.size[0]
    y = rect[1] / img.size[1]
    w = rect[2] / img.size[0]
    h = rect[3] / img.size[1]
    img = (np.array(img.resize((size, size)))/128.0 - 1).astype(np.float32).T
    r = np.array([x, y, w, h], np.float32)
    imgX.append(img)
    rectY.append(r)

imgX = np.array(imgX)
rectY = np.array(rectY)
rectY = (2 * rectY - 1).astype(np.float32)

index = np.random.permutation(len(imgX))
train_index = index[:int(len(index)*0.8)]
test_index = index[int(len(index)*0.8):]



"""predicted_rect = np.zeros(rectY[test_index].shape)
for i in range(4):
    print('start fit %s'%i)
    L = LinearRegression()
    L.fit(imgX[train_index].reshape(len(train_index), -1), rectY[train_index, [i]])

    print(L.score(imgX[train_index].reshape(len(train_index), -1), rectY[train_index, [i]]))
    print(L.score(imgX[test_index].reshape(len(test_index), -1), rectY[test_index, [i]]))
    predicted_rect[:, i] = L.predict(imgX[test_index].reshape(len(test_index), -1))"""


model = FaceNet()
print('start fit')
learning_curve = model.fit(imgX[train_index], rectY[train_index], 20, 10)
print('end fit')
print(model.score(imgX[train_index], rectY[train_index]))
print(model.score(imgX[test_index], rectY[test_index]))

predicted_rect = model.predict(imgX[train_index])

pl.figure()
pl.plot(learning_curve)
pl.show()

rectY = 0.5 * (rectY + 1)
predicted_rect = 0.5 * (predicted_rect + 1)



for img, rect, rect_true in zip(imgX[test_index], predicted_rect, rectY[test_index]):
    image = (img.T + 1)/2
    fig, ax = pl.subplots(1)
    ax.imshow(image)
    rect_patch_predicted = patches.Rectangle((rect[0]*size, rect[1]*size), rect[2] * size, rect[3] * size,
                                   linewidth=1, edgecolor='r', facecolor='none')
    rect_patch_true = patches.Rectangle((rect_true[0] * size, rect_true[1] * size), rect_true[2] * size, rect_true[3] * size,
                                             linewidth=1, edgecolor='g', facecolor='none')

    ax.add_patch(rect_patch_predicted)
    ax.add_patch(rect_patch_true)
    pl.show()



