# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from time import sleep

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    fig = plt.figure()

    x = np.linspace(0, 5, 10)
    y = x ** 2
    # This creates the size of the figure  [left, bottom, width, height (range 0 to 1)]
    axes1 = fig.add_axes([0, 0, 1, 1])  # main axes
    axes1.plot(x, y, 'red')
    axes1.set_xlabel('x')
    axes1.set_ylabel('y')
    axes1.set_title('title')

    # plt.pause(1)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    sleep(0.3)

    axes1.clear()
    axes1.plot(y, x, 'blue')
    axes1.set_xlabel('x')
    axes1.set_ylabel('y')
    axes1.set_title('title')

    display.display(plt.gcf())
    display.clear_output(wait=True)
    sleep(0.3)

    axes1.clear()
    axes1.plot(x, y, 'red')
    axes1.set_xlabel('x')
    axes1.set_ylabel('y')
    axes1.set_title('title')

    # plt.pause(1)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    sleep(0.3)

    axes1.clear()
    axes1.plot(y, x, 'blue')
    axes1.set_xlabel('x')
    axes1.set_ylabel('y')
    axes1.set_title('title')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
