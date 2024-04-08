import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from IPython import display

plt.ion()

def plot(train_loss, val_loss = [0], save = None, line = None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.title('Training...')
    plt.xlabel('Epochs')
    plt.ylabel(line)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.tick_params(axis='x', which='major')
    plt.ylim(ymin=0)
    plt.text(len(train_loss)-1, train_loss[-1], str(train_loss[-1]))
    plt.text(len(val_loss)-1, val_loss[-1], str(val_loss[-1]))
    plt.savefig(save)
    plt.show(block=False)
    plt.pause(1)
    plt.close("all")