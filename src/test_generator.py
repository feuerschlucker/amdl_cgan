import numpy as np
import caltechFN_cgan_v1
# Test the generator
latent_dim = 100
n_classes = 10
generator = define_generator(latent_dim, n_classes)

# Generate a sample image
latent_vector = np.random.randn(1, latent_dim)  # Example latent vector
label = np.array([1])  # Example label
generated_image = generator.predict([latent_vector, label])

print("Generated image shape:", generated_image.shape)  # Should be (1, 28, 50, 3)
