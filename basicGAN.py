import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time,os
import datetime
import tensorflow
import json

# Load MNIST dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Get the current date and time
current_datetime = datetime.datetime.now()
dt =current_datetime.strftime("%d-%m-%Y_%H-%M")


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Build the generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return model

generator = make_generator_model()


# build discriminator model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    
    return model

discriminator = make_discriminator_model()

# define loss and optimizers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# define training



EPOCHS = 2
noise_dim = 100
num_examples_to_generate = 16

# Seed for consistent images across epochs
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Initialize lists to store losses
generator_losses = []
discriminator_losses = []


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            # Append the losses to the lists
            generator_losses.append(gen_loss.numpy())
            discriminator_losses.append(disc_loss.numpy())


        # Produce images for the GIF
        generate_and_save_images(generator, epoch + 1, seed)

        print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec')

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    
# path to store generated images
path = f'/Users/yogesh/pythoncode/GenerativeAI/BasicGANData/{dt}_{EPOCHS}/'
os.makedirs(path,exist_ok=True)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'{path}image_at_epoch_{epoch:04d}.png')
    #plt.show()



# Train GAN
train(train_dataset, EPOCHS)

# save Generator loss and Discriminator loss to json
generator_losses = [float(loss) for loss in generator_losses]
discriminator_losses = [float(loss) for loss in discriminator_losses]
loss_data = {
    "Generator_loss" : generator_losses,
    "Discriminator_loss": discriminator_losses
}

# write a disctionary to json file
json_path = f'{path}/loss_data.json'
with open(json_path,'w')as json_file : 
    json.dump(loss_data,json_file)
print("Loss data file exported successfully.")
    
# plot the Generator loss and Discriminator loss
plt.figure(figsize=(8, 6))

# Plotting the discriminator loss
plt.plot(discriminator_losses, label='Discriminator loss', color='blue', alpha=0.7)
# Plotting the generator loss
plt.plot(generator_losses, label='Generator loss', color='orange', alpha=0.7)

# Adding titles and labels
plt.title('Generator and Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Adding a legend
plt.legend()
# Show the plot
plt.show()