import matplotlib.pyplot as plt

def my_plot(file_path):
    epochs = []
    train_loss = []
    
    train_acc = []
    train_prec = []
    train_rec = []
    train_f1 = []
    
    test_loss = []
    
    test_acc = []
    test_prec = []
    test_rec = []
    test_f1 = []
    
    train_classif_loss = []
    test_classif_loss = []
    train_rep_loss = []
    test_rep_loss = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            line = line.replace("'", '')
            if line:  # Ignore empty lines
                values = list(map(float, line.strip('[]').split(',')))
                values = [round(v, 3) for v in values]  # Round all values to 3 decimal places
                
                # Epoch
                epochs.append(int(values[0]))
                
                # acc, prec, rec, f1, avg_loss, rep_loss, classif_loss
                
                # Train metrics
                train_acc.append(values[1])
                train_prec.append(values[2])
                train_rec.append(values[3])
                train_f1.append(values[4])
                
                # Train loss
                train_loss.append(values[5])
                train_rep_loss.append(values[6])
                train_classif_loss.append(values[7])
                
                # acc, prec, rec, f1, avg_loss, rep_loss, classif_loss
                
                # Test metrics
                test_acc.append(values[8])
                test_prec.append(values[9])
                test_rec.append(values[10])
                test_f1.append(values[11])
                
                # Test loss
                test_loss.append(values[12])
                test_rep_loss.append(values[13])
                test_classif_loss.append(values[14])
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    
    def plotter(x,y, xval, yval, ylabel, title):
        axs[x, y].plot(epochs, xval, 'r', label='Train')
        axs[x, y].plot(epochs, yval, 'b', label='Test')
        axs[x, y].set_xlabel('Epoch')
        axs[x, y].set_ylabel(ylabel)
        axs[x, y].set_title(title)
        axs[x, y].legend()
        axs[x, y].grid(True)
    
    # Plot Loss
    plotter(0, 0, train_loss, test_loss, 'Loss', 'Total Loss')
    plotter(0, 1, train_rep_loss, test_rep_loss, 'Loss', 'Representation Loss')
    plotter(0, 2, train_classif_loss, test_classif_loss, 'Loss','Classification Loss')

    # Plot Metrics (Accuracy, Precision, Recall, F1)
    plotter(1, 0, train_acc, test_acc, 'Accuracy (%)', 'Accuracy')
    plotter(1, 1, train_prec, test_prec, 'Precision (%)', 'Precision')
    plotter(1, 2, train_rec, test_rec, 'Recall (%)', 'Recall')
    plotter(2, 0, train_f1, test_f1, 'F1 Score (%)', 'F1 Score')
    
    # Hide the empty subplot
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    
    plt.ioff()
    plt.tight_layout()
    fig.savefig('saved_models/all_plots.png')
    plt.close('all')


if __name__ == "__main__":
    my_plot('../saved_models/log.txt')
