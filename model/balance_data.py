import numpy as np
import cv2
import pandas as pd
from random import shuffle
from collections import Counter


def balance_data():
    train_data = np.load('training_data.npy')

    df = pd.DataFrame(train_data)
    print(Counter(df[1].apply(str)))

    rotate = []
    left = []
    right = []

    shuffle(train_data)

    for data in train_data:
        img = data[0]
        choice = data[1]

        if choice == [1, 0, 0]:
            rotate.append([img, choice])
        elif choice == [0, 1, 0]:
            left.append([img, choice])
        elif choice == [0, 0, 1]:
            right.append([img, choice])

    # making the entries the same size
    #left = left[:500]
    #right = right[:500]
    #rotate = rotate[:500]

    final_data = rotate + left + right

    shuffle(final_data)
    print(len(final_data))
    np.save('final_data.npy', final_data)


train_data = np.load('training_data.npy')
df = pd.DataFrame(train_data)
print(Counter(df[1].apply(str).sort_values(ascending=True)))
print(Counter(df[1].apply(str)[0]))
