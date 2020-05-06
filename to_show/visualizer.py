import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import os




def main():
    directory = os.fsencode("./saved/")

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print(filename)
            logs = np.loadtxt("./saved/"+filename, delimiter = ',')
            a, b, c = logs[0], logs[1], logs[2]
            #print (logs, logs.shape)
            n = len(a)
            x = range(1,n+1)
            xtick = range(0, n+2, n//10)
            plot_fig(a,b,c, filename,x, xtick)


##
def plot_fig(a,b,c,filename,x, xtick):
    plt.clf()
    plt.plot(x,a,x,b,x,c)
    plt.legend(('training loss', 'testing loss', 'accuracy'))
    plt.xticks(xtick)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss/accuracy')
    plt.savefig(f'./figs/{filename[:-4]}.png', dpi=300)




if __name__ == "__main__":
    main()