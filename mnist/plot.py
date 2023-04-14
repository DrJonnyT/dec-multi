import matplotlib.pyplot as plt

                        
def plot_mnist_10x10(images, labels, suptitle):
    """
    Make a 10x10 plot of MNIST digits

    Parameters
    ----------
    images : array of images
        MNIST images
    labels : array of int
        MNIST labels corresponding to X
    suptitle : string
        Suptitle of the plot

    Returns
    -------
    None

    """
    # Create a dictionary to store 10 examples of each digit
    digit_dict = {}
    for i in range(10):
        digit_dict[i] = []

    # Add 10 examples of each digit to the dictionary
    for image, label in zip(images, labels):
        if len(digit_dict[label]) < 10:
            digit_dict[label].append(image)

    # Plot the 10 examples of each digit
    fig, axs = plt.subplots(10, 10, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle(suptitle,fontsize=20)

    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(digit_dict[j][i].reshape(28,28), cmap=plt.cm.binary)
            axs[i, j].axis('off')

    return fig, axs