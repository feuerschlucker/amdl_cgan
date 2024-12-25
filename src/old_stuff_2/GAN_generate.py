from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot as plt


def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input



def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
        
    plt.savefig('output250.png')
    #plt.show()

model = load_model('cifar_generator_250epochs.keras') #Model trained for 100 epochs
latent_points = generate_latent_points(100, 25)  #Latent dim and n_samples
X = model.predict(latent_points)
X = (X + 1) / 2.0
import numpy as np
X = (X*255).astype(np.uint8)
show_plot(X, 5)