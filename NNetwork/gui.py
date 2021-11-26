from tkinter import *
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
import random

from PIL import ImageGrab, Image, ImageOps
from network import Network as Model

from mnist_loader import mnist_loader


window = Tk()
window.geometry('700x550')
window.resizable(0, 0)
window.title('Handwritten digit recognition')

cv = Canvas(window, width=500, height=500, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

l1 = Label()

lastx, lasty = None, None

training_data, validation_data, test_data = mnist_loader
net = Model.load('hardly_regularized.json')

print(f'Test network:\n\tAccuracy on test: {net.accuracy(test_data)}')

image_number = 0


def recognize():
    global l1, net, image_number
    widget = cv

    # find coordinates to make screen
    x = window.winfo_rootx() + widget.winfo_x() + 4
    y = window.winfo_rooty() + widget.winfo_y() + 4
    width = widget.winfo_width() * 2 - 15
    height = widget.winfo_height() * 2 - 15
    img_screen = pyautogui.screenshot(region=(x*2, y*2, width, height))

    # save screen to file
    image_path = f'images/drawn_{image_number}.png'
    img_screen.save(image_path)

    image = Image.open(image_path)

    # convert it to grey scale
    image = image.convert('L')

    # invert black-> white, white-> black
    image = ImageOps.invert(image)

    # resize
    image = image.resize((28, 28))
    image_arr = np.array(image.getdata()).reshape((784, 1))

    # convert all black to not so black
    image_arr[image_arr > 50] = 255 * random.uniform(0.7, 1)

    # make white background
    image_arr[image_arr < 50] = 0

    # save to see how its looks for our network
    processed_path = f'images/processed_{image_number}.png'
    image.save(processed_path)

    # see result online
    plt.imshow(image_arr.reshape((28, 28)), cmap=plt.cm.binary)
    plt.show()

    # resize to dump it network
    image_arr = [pixel / 255 for pixel in image_arr]

    # predict
    predicted_seq = net.feedforward(image_arr)

    # sort results to see best
    pred_num = [(index, i) for index, i in enumerate(predicted_seq)]
    pred_best = sorted(pred_num, key=lambda x: x[1], reverse=True)

    # best result
    predicted = pred_best[0]

    l1 = Label(window, text=f"1.Predicted = {predicted[0]}\nValue: {predicted[1]}\n\n"
                            f"2.Predicted = {pred_best[1][0]}\nValue: {pred_best[1][1]}\n"
                            f"3.Predicted = {pred_best[2][0]}\nValue: {pred_best[2][1]}\n", font=('Algrerian', 20))
    l1.grid(row=0, column=3, pady=1, padx=1)
    image_number += 1
    return


def clear_widget():
    global cv, l1
    cv.delete('all')
    l1.destroy()


def event_activation(event):

    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=35, fill='black', capstyle=ROUND, smooth=TRUE)
    lastx, lasty = x, y


b1 = Button(window, text='1. Clear Canvas', font=('Algerian', 15), bg='orange', fg='black', command=clear_widget)
b1.grid(row=2, column=0, pady=1, padx=1)

b2 = Button(window, text='2. Prediction', font=('Algerian', 15), bg='white', fg='red', command=recognize)
b2.grid(row=2, column=1, pady=1, padx=1)

cv.bind('<Button-1>', event_activation)
window.mainloop()
