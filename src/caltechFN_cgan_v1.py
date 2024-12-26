import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate

from matplotlib import pyplot as plt
########################################################################
def showsamples():
    (trainX, trainy) = load_real_samples()

    # plot 25 images
    for i in range(25):
        plt.subplot(5, 5, 1 + i)
        plt.axis('off')
        plt.imshow(trainX[i])
    plt.savefig('plots/25_FNsamples.png')


def define_discriminator(in_shape=(50, 28, 3), n_classes=10):
    # Label input
    in_label = Input(shape=(1,))  # Shape (batch_size, 1)
    li = Embedding(n_classes, 50)(in_label)  # Shape (batch_size, 1, 50)
    li = Flatten()(li)  # Flatten to (batch_size, 50)
    li = Dense(in_shape[0] * in_shape[1] * in_shape[2])(li)  # Shape (batch_size, 50*28*3)
    li = Reshape((in_shape[0], in_shape[1], in_shape[2]))(li)  # Reshape to (batch_size, 50, 28, 3)

    # Image input
    in_image = Input(shape=in_shape)  # Shape (50, 28, 3)

    # Combine image and label
    merge = Concatenate()([in_image, li])  # Shape (50, 28, 6)

    # Downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)  # Shape (25, 14, 128)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)  # Shape (13, 7, 128)
    fe = LeakyReLU(alpha=0.2)(fe)

    # Flatten and classify
    fe = Flatten()(fe)  # Shape (batch_size, 11648)
    fe = Dropout(0.4)(fe)
    out_layer = Dense(1, activation='sigmoid')(fe)  # Shape (batch_size, 1)

    # Define the model
    model = Model([in_image, in_label], out_layer)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model



def define_generator(latent_dim, n_classes=10):
    # Label input
    in_label = Input(shape=(1,))  # Shape (batch_size, 1)
    li = Embedding(n_classes, 50)(in_label)  # Shape (batch_size, 1, 50)
    n_nodes = 25 * 14  # Adjust dimensions to align with target size
    li = Dense(n_nodes)(li)  # Shape = (batch_size, 350)
    li = Reshape((25, 14, 1))(li)  # Reshape to (batch_size, 25, 14, 1)

    # Latent vector input
    in_lat = Input(shape=(latent_dim,))  # Shape (batch_size, latent_dim)

    n_nodes = 128 * 25 * 14  # Adjust to match the intermediate dimensions
    gen = Dense(n_nodes)(in_lat)  # Shape (batch_size, 128 * 25 * 14)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((25, 14, 128))(gen)  # Shape (batch_size, 25, 14, 128)

    # Concatenate label embedding with generated image
    merge = Concatenate()([gen, li])  # Shape (batch_size, 25, 14, 129)

    # Upsample and generate the image
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)  # Shape (batch_size, 50, 28, 128)
    gen = LeakyReLU(alpha=0.2)(gen)

    out_layer = Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')(gen)  # Shape (batch_size, 50, 28, 3)

    # Define the model
    model = Model([in_lat, in_label], out_layer)
    return model



test_gen = define_generator(100, n_classes=10)
print(test_gen.summary())

def define_gan(g_model, d_model):
	d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.
	gen_noise, gen_label = g_model.input  #Latent vector size and label size
	gen_output = g_model.output  #32x32x3
	gan_output = d_model([gen_output, gen_label])
	model = Model([gen_noise, gen_label], gan_output)
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


def load_real_samples():
    # Load the preprocessed dataset
    data = np.load('processed_dataset.npz')
    images = data['images']
    labels = data['labels']
    return [images, labels]

def generate_real_samples(dataset, n_samples):

	images, labels = dataset  
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix]
	y = ones((n_samples, 1))  #Label=1 indicating they are real
	return [X, labels], y

def generate_latent_points(latent_dim, n_samples, n_classes=10):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	images = generator.predict([z_input, labels_input])
	y = zeros((n_samples, 1))  #Label=0 indicating they are fake
	return [images, labels_input], y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)  #the discriminator model is updated for a half batch of real samples 
	for i in range(n_epochs):
		for j in range(bat_per_epo):
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			y_gan = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
	g_model.save('cifar_conditional_generator_500epochs.keras')



def main():
    # latent_dim = 100
    # d_model = define_discriminator()
    # print("Discriminator input shape:", d_model.input_shape)  # Should match (28, 50, 3)
    # print(d_model.summary())
    # g_model = define_generator(latent_dim)
    # z_input, labels_input = generate_latent_points(latent_dim, 1)
    # generated_image = g_model.predict([z_input, labels_input])
    # print("Generated image shape:", generated_image.shape)  # Should be (1, 28, 50, 3)
    # gan_model = define_gan(g_model, d_model)
    # dataset = load_real_samples()
    # print("Dataset images shape:", dataset[0].shape)  # Should be (num_samples, 28, 50, 3)
    # print("Dataset labels shape:", dataset[1].shape)  # Should match one-hot encoding

    # train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1)
    showsamples()

    print("main")
    pass

if __name__ == '__main__':
	main()