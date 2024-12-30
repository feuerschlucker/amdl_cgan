import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import time
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
from keras.models import Sequential
from keras.regularizers import l2
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


def define_discriminator(in_shape=(28, 50, 3), n_classes=10):
    # Label input
    in_label = Input(shape=(1,))  # Shape (batch_size, 1)
    li = Embedding(n_classes, 50)(in_label)  # Shape (batch_size, 1, 50)
    
    n_nodes = in_shape[0] * in_shape[1]  # 28x50 = 1400.
    li = Dense(n_nodes)(li)  # Shape = (batch_size, 1400)
    li = LeakyReLU(alpha=0.2)(li)
    
    li = Reshape((in_shape[0], in_shape[1], 1))(li)  # Reshape to (batch_size, 28, 50, 1)

    # Image input
    in_image = Input(shape=in_shape)  # Shape (28, 50, 3)

    # Combine image and label
    merge = Concatenate()([in_image, li])  # Shape (batch_size, 28, 50, 4)

    # Downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)  # Shape (14, 25, 128)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)  # Shape (7, 13, 128)
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
    label_input = Input(shape=(1,))
    li = Embedding(n_classes, 50)(label_input)  # Shape (batch_size, 1, 50)
    li = Dense(14 * 25 * 1)(li)  # Shape (batch_size, 1, 14 * 25 * 1)
    li = LeakyReLU(alpha=0.2)(li)
    li = Reshape((14, 25, 1))(li)  # Reshape to (batch_size, 14, 25, 1)

    # Latent vector input
    latent_input = Input(shape=(latent_dim,))
    gen = Dense(14 * 25 * 128)(latent_input)  # Shape (batch_size, 14 * 25 * 128)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((14, 25, 128))(gen)  # Reshape to (batch_size, 14, 25, 128)

    # Combine latent vector and label embedding
    merge = Concatenate()([gen, li])  # Shape (batch_size, 14, 25, 129)

    # Add generator layers using Sequential-like API
    generator = Sequential([
        Input(shape=(14, 25, 129)),  # Input from concatenation
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',kernel_regularizer=l2(0.01)),  # Shape (28, 50, 128)
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', activation='tanh')  # Shape (28, 50, 3)
    ])

    # Wrap the combined model
    combined_output = generator(merge)
    model = Model(inputs=[latent_input, label_input], outputs=combined_output)

    return model





def define_gan(g_model, d_model):
	d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.
	gen_noise, gen_label = g_model.input  #Latent vector size and label size
	gen_output = g_model.output  #28x50x3
	gan_output = d_model([gen_output, gen_label])
	model = Model([gen_noise, gen_label], gan_output)
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


def load_real_samples():
    # Load the preprocessed dataset
    data = np.load('balanced_dataset.npz')
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



def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=5, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    # To store losses
    d_real_loss_hist, d_fake_loss_hist, g_loss_hist = [], [], []
    train_generator_steps = 2
    for i in range(n_epochs):
        d_real_loss, d_fake_loss, g_loss = 0, 0, 0  # Track cumulative losses for the epoch

        for j in range(bat_per_epo):
            # Generate real samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            d_real_loss += d_loss_real  # Accumulate real loss

            # Generate fake samples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            d_fake_loss += d_loss_fake   # Accumulate fake loss

            for _ in range(train_generator_steps):  # Train the generator multiple times
                # Generate latent points for the generator
                [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
                y_gan = ones((n_batch, 1))  # Labels to trick the discriminator
                g_loss_batch = gan_model.train_on_batch([z_input, labels_input], y_gan)
                g_loss += g_loss_batch  # Accumulate generator losses

            print(f'Epoch : {i+1} , Batch : {j+1}')

        # Calculate average losses for the epoch
        d_real_loss /= bat_per_epo
        d_fake_loss /= bat_per_epo
        g_loss /= (bat_per_epo * train_generator_steps)

        # Save losses
        d_real_loss_hist.append(d_real_loss)
        d_fake_loss_hist.append(d_fake_loss)
        g_loss_hist.append(g_loss)

        print(f'Epoch>{i+1}, d1={d_real_loss:.3f}, d2={d_fake_loss:.3f}, g={g_loss:.3f}')

    # Save the generator model
    g_model.save('models/FN_balanced_30epochs_reg.keras')

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(d_real_loss_hist, label='Discriminator Real Loss')
    plt.plot(d_fake_loss_hist, label='Discriminator Fake Loss')
    plt.plot(g_loss_hist, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Training')
    plt.legend()
    plt.savefig('plots/loss_plot_30epochs_reg.png')
    plt.show()
    

def main():
    latent_dim = 100
    d_model = define_discriminator()
    print("Discriminator input shape:", d_model.input_shape)  # Should match (28, 50, 3)
    print(d_model.summary())
    g_model = define_generator(latent_dim)
    z_input, labels_input = generate_latent_points(latent_dim, 1)
    generated_image = g_model.predict([z_input, labels_input])
    print("Generated image shape:", generated_image.shape)  # Should be (1, 28, 50, 3)
    gan_model = define_gan(g_model, d_model)
    dataset = load_real_samples()
    print("Dataset images shape:", dataset[0].shape)  # Should be (num_samples, 28, 50, 3)
    print("Dataset labels shape:", dataset[1].shape)  
    t1=time.time()
    train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=30)
    print(time.time()-t1)
    #showsamples()

    print("main")
    pass

if __name__ == '__main__':
	main()