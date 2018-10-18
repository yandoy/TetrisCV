import numpy as np
import cv2
import time
import os
import keyboard
import tensorflow as tf
from score import *
from balance_data import *
from train_model import *
from getkeys import key_pressed, keys_to_output
from PIL import ImageChops
from grabscreen import grab_screen
from load_model import *
from alexnet import alexnet
from keras import backend as K


def rotate():
    keyboard.press_and_release('up')


def move_left():
    keyboard.press_and_release('left')


def move_right():
    keyboard.press_and_release('right')


def down():
    keyboard.press_and_release('down')


def translate_move(moves):
    if moves == [1, 0, 0, 0]:
        rotate()
    elif moves == [0, 1, 0, 0]:
        down()
    elif moves == [0, 0, 1, 0]:
        move_left()
    elif moves == [0, 0, 0, 1]:
        move_right()


load_model()


graph = tf.get_default_graph()


def main():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    game_over = np.load('end_screen.npy')
    paused = False
    filename = "training_data.npy"
    training_data = []
    screen_old = grab_screen(region=(510, 860, 640, 920))
    screen_old = cv2.cvtColor(screen_old, cv2.COLOR_BGR2GRAY)

    while True:
        if not paused:
            screen_game = grab_screen(region=(696, 420, 970, 950))
            screen_game = cv2.cvtColor(screen_game, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen_game, (60, 60))

            # making moves
            with graph.as_default():
                prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
                moves = list(np.around(prediction))
                translate_move(moves)

            # recording moves
            keys = key_pressed()
            output = keys_to_output(keys)
            training_data.append([screen, output])

            screen_end = grab_screen(region=(600, 400, 620, 420))
            screen_end = cv2.cvtColor(screen_end, cv2.COLOR_BGR2GRAY)

            screen_score = grab_screen(region=(510, 860, 640, 920))
            screen_score = cv2.cvtColor(screen_score, cv2.COLOR_BGR2GRAY)

            if (screen_old != screen_score).any():
                increase_score()
                screen_old = screen_score

            if (screen_end == game_over).all():
                if check_score() > check_high_score() and check_score() > 2:
                    set_high_score(check_score())
                    K.clear_session()
                    np.save(filename, training_data)
                    # here we will do the re-training
                    balance_data()
                    train_model()
                time.sleep(0.5)
                keyboard.press_and_release('enter')
                keyboard.press_and_release('enter')
                print(check_score())
                training_data = []
                reset_score()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        if keyboard.is_pressed('p'):
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True


main()


