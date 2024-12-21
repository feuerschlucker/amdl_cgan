import json
import matplotlib.pyplot as plt
from matplotlib import cm

def load_files():
    #files = [ 'bigfirst.json', 'smallfirst.json', 'big_model.json', 'bm_2.json']
    #labels = ['bigfirst', 'smallfirst', 'big_gmodel', 'bm_2']
    files = [ 'm1.json','m2.json','m3.json','m4.json','m5.json','m6.json','m7.json','m8.json','m9.json','m10.json','m11.json']
    labels = ['m1,','m2','m3','m4','m5' ,'m6','m7','m8','m9','m10','m11', ]
    files = [ 'm11.json','m12.json','m13.json']#,'m4.json','m5.json','m6.json','m7.json','m8.json','m9.json','m10.json','m11.json']
    labels = ['4_layer_reg_lim', '5_layer_droput_only', '5_layer_reg_lim']#,'m4','m5' ,'m6','m7','m8','m9','m10','m11', ]
    history = []

    fig, (ax1,ax2) = plt.subplots(2,figsize=(8, 10), sharex=True)
    
    
    #ax2 = ax1.twinx()  # Create the secondary y-axis outside the loop
    cmap = cm.get_cmap('tab10', len(files))
    cmap1 = cm.get_cmap('Dark2', len(files))

    for j in range(len(files)):
        i = len(files) - 1 - j
        with open(f"data/{files[i]}", 'r') as file:
            history.append(json.load(file))
        print("epochs :  ",len(history[j]['loss']))
        # Plot loss on the primary y-axis
        ax1.plot(history[j]['loss'], label=f'{labels[i]} Tr.', color=cmap(i),linestyle='--' ,linewidth=2)
        ax1.plot(history[j]['val_loss'], label=f'{labels[i]} Val.', color=cmap(i),  linewidth=2)
        
        # Plot accuracy on the secondary y-axis
        ax2.plot(history[j]['accuracy'], label=f'{labels[i]} Tr.', color=cmap(i),linestyle='--',linewidth=2)
        ax2.plot(history[j]['val_accuracy'], label=f'{labels[i]} Val.', color=cmap(i),  linewidth=2)

    # Set labels and titles
    ax2.set_ylim(0, 1)  # Typical range for accuracy (0 to 1)
    ax1.set_xlim(0, 125)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax2.set_ylabel('Accuracy', color='black')
    ax1.set_title('Training and Validation Data, Loss and Accuracy')

    # Add legends for both axes
    ax1.legend(loc='upper right', ncol=3)
    ax2.legend(loc='lower right',ncol=3)

    # Save and display the plot
    plt.savefig('plots/m111213xx.png')
    plt.show()

def main():
    load_files()

if __name__ == "__main__":
    main()
