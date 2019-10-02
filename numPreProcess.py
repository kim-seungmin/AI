import cv2
import os
import numpy as np
import math

path_dir = './testNumber/'
f_list = os.listdir(path_dir)
file_list = [file for file in f_list if file.endswith(".jpg")]
file_list.sort()

for file in file_list:
    fname = path_dir+file
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    (thresh, gray) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    #shiftx, shifty = getBestShift(gray)
    #shifted = shift(gray, shiftx, shifty)
    #gray = shifted
    new_fname = fname = path_dir+'a/'+file
    cv2.imwrite(new_fname, gray, None)

    cv2.waitKey(0)

cv2.destroyAllWindows()