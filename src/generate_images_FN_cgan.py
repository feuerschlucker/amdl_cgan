##########################################################
# Now, let us load the generator model and generate images
# Lod the trained model and generate a few images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import cv2
# 

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	# labels = randint(0, n_classes, n_samples)
	return z_input #, labels]


# plot the result (10 sets of images, all images in a column should be of same class in the plot)
# Plot generated images 
def show_plot(examples, n):
	plt.figure(figsize=(13, 4))
	for i in range(30):
		plt.subplot(3, 10, 1 + i)
		plt.axis('off')
		plt.imshow(cv2.transpose(examples[i, :, :, :]))
	plt.tight_layout()
	plt.savefig('plots/FN_balanced_100epochs.png')
    


def main():
    # load model
	model = load_model('models/FN_balanced_100epochs.keras')
	latent_points = generate_latent_points(100,30)
	#print(labels)
	labels = asarray([x for _ in range(3) for x in range(10)])
	print(labels)
	X  = model.predict([latent_points, labels])
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	X = (X*255).astype(np.uint8)
	show_plot(X, 4)

if __name__ == '__main__':
	main()