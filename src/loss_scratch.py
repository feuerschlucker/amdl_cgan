import matplotlib.pyplot as plt

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    # To store losses
    d_real_loss_hist, d_fake_loss_hist, g_loss_hist = [], [], []

    for i in range(n_epochs):
        d_real_loss, d_fake_loss, g_loss = 0, 0, 0  # Track cumulative losses for the epoch

        for j in range(bat_per_epo):
            # Generate real samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            # Generate fake samples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)

            # Generate latent points for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss_batch = gan_model.train_on_batch([z_input, labels_input], y_gan)

            # Aggregate losses
            d_real_loss += d_loss_real
            d_fake_loss += d_loss_fake
            g_loss += g_loss_batch

        # Calculate average losses for the epoch
        d_real_loss /= bat_per_epo
        d_fake_loss /= bat_per_epo
        g_loss /= bat_per_epo

        # Save losses
        d_real_loss_hist.append(d_real_loss)
        d_fake_loss_hist.append(d_fake_loss)
        g_loss_hist.append(g_loss)

        print(f'Epoch>{i+1}, d1={d_real_loss:.3f}, d2={d_fake_loss:.3f}, g={g_loss:.3f}')

    # Save the generator model
    g_model.save('models/FN_1epochs.keras')

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(d_real_loss_hist, label='Discriminator Real Loss')
    plt.plot(d_fake_loss_hist, label='Discriminator Fake Loss')
    plt.plot(g_loss_hist, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Training')
    plt.legend()
    plt.savefig('plots/loss_plot.png')
    plt.show()
