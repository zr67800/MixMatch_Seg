import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import os


x = range(1,201)
xtick = range(0,202,20)

def main():
    directory = os.fsencode("./saved/")
    s = []
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print(filename[0])
            logs = np.loadtxt("./saved/"+filename, delimiter = ',')
            a, b, c = logs[0], logs[1], logs[2]
            #print (logs, logs.shape)
            #plot_fig(a,b,c, filename)
            if filename[0] == "0":
                s0 = b
            elif filename[0] == "8":
                s8 = b
            elif filename[0] == "1":
                s1 = b
            elif filename[0] == "2":
                s2 = b
            else:
                s.append(b)
    
    for si in s:
        plt.plot(x,si,'g', linewidth = 0.5)
    plt.plot(x,s1,'y', linewidth = 0.5)
    plt.plot(x,s2,'c', linewidth = 0.5)
    plt.plot(x,s0,'r', linewidth = 0.5)
    plt.plot(x,s8,'b', linewidth = 0.5)

    plt.xticks(xtick)
    plt.ylim(-0.1, 1.1)
    plt.savefig('./figs/comp.png', dpi=300)


##
def plot_fig(a,b,c,filename):
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