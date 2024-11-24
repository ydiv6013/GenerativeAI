
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time ,json
from tqdm import tqdm
import datetime
size = 64
# Hyperparameters
EPOCHS = 500
BATCH_SIZE = 64
noise_dim = 128
lambda_gp = 10


# load and preprocess dataset

dataset_path = '/Users/yogesh/pythoncode/datasets/img_align_celeba/img_align_celeba'

def load_dataset(image_size = (size,size),batch_size = BATCH_SIZE,dataset_dir = dataset_path):
    # GANs typically work best when image values are in the range [-1, 1]
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./127.5 - 1) # Normalize to [-1, 1]
    dataset = datagen.flow_from_directory(dataset_dir,
                                          target_size=image_size,
                                          batch_size=batch_size,
                                          classes=['.'],  # Treat everything as part of one single "class"
                                          class_mode=None,
                                          shuffle=True)
    return dataset



# Build the generator model
def make_generator_model():
    generator_model = tf.keras.Sequential([
        tf.keras.layers.Dense(4*4*512, use_bias=False, input_shape=(noise_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((4, 4, 512)),
        # Upsample to (8, 8)
        tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # Upsample to (16, 16)
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # Upsample to (32, 32)
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # Upsample to (64, 64)
        tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ])
    return generator_model
gm = make_generator_model()
gm.summary()
# Build the discriminator model
def make_discriminator_model():
    discriminator_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return discriminator_model

# Define the Wasserstein loss and gradient penalty
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator,real_image,fake_image):
    batch_size = tf.shape(real_image)[0]
    alpha =tf.random.uniform([batch_size,1,1,1],0.0,1.0)
    interpolated_images = alpha * real_image + (1-alpha) * fake_image
    with tf.GradientTape() as tape :
        tape.watch(interpolated_images)
        interpolated_scores = discriminator(interpolated_images,training = True)
    gradients = tape.gradient(interpolated_scores, interpolated_images)
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty


# Define the training step
@tf.function 

def train_step(images,generator,discriminator,generator_optimizer,discriminator_optimizer,lambda_gp):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size,noise_dim])
    
    with tf.GradientTape() as gen_tape , tf.GradientTape() as disc_tape : 
        generated_images = generator(noise,training = True)
        print(f"Generated images shape: {generated_images.shape}")
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = wasserstein_loss(tf.ones_like(fake_output), fake_output)
        disc_loss = wasserstein_loss(tf.ones_like(real_output), real_output) - wasserstein_loss(tf.zeros_like(fake_output), fake_output)
        gp = gradient_penalty(discriminator, images, generated_images)
        disc_loss += lambda_gp * gp
        
    
    # Compute and apply gradients with clipping
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Clip gradients before applying
    gradients_of_generator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_generator]
    gradients_of_discriminator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_discriminator]

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    
    return gen_loss, disc_loss

gen_losses, disc_losses = [],[]

# Define the training loop with multiple discriminator updates
def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, lambda_gp, discriminator_updates_per_generator=1):
    
    for epoch in range(epochs):
        start = time.time()

        # Training loop
        for image_batch in tqdm(dataset, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
            # Train the discriminator multiple times
            for _ in range(discriminator_updates_per_generator):
                gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, lambda_gp)
                disc_losses.append(disc_loss.numpy())
                
            # Train the generator once after updating the discriminator multiple times
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            with tf.GradientTape() as gen_tape:
                generated_images = generator(noise, training=True)
                fake_output = discriminator(generated_images, training=True)
                gen_loss = wasserstein_loss(tf.ones_like(fake_output), fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            gen_losses.append(gen_loss.numpy())
        print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec')
        print(f'Epoch {epoch + 1}, Generator loss: {gen_loss.numpy()}, Discriminator loss: {disc_loss.numpy()}')

        # Generate and save images after each epoch
        generate_and_save_images(generator, epoch + 1, seed)
        # Visualize a sample of generated images using the generate_sample_images function
        #generate_sample_images(generator,epoch + 1, seed)

    generate_and_save_images(generator, epochs, seed)
    
# Get the current date and time
current_datetime = datetime.datetime.now()
dt =current_datetime.strftime("%d-%m-%Y_%H-%M")
    
path = f'/Users/yogesh/pythoncode/GenerativeAI/AdvancedGANData/{dt}'
os.makedirs(path,exist_ok=True)

# Generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2.0)
        plt.axis('off')
        
    plt.savefig(f'{path}/image_at_epoch_{epoch:04d}.png')
    plt.close()

def generate_sample_images(model,epoch, test_input):
    predictions = model(test_input, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow((predictions[i] + 1) / 2.0)
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.show()



# Create models
generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)

# Load dataset
train_dataset = load_dataset((size,size),BATCH_SIZE,dataset_path)

print(f"Total Train Batches : {len(train_dataset)}")

# Limit  batches
max_batches = int(input("Enter Number of batches to train : "))
batch_count = 0

train_dataset_sliced = []
# Iterate through the dataset
for image_batch in train_dataset:
    #print(f"Batch {batch_count + 1}: {image_batch.shape}")
    batch_count += 1
    train_dataset_sliced.append(image_batch)
    if batch_count >= max_batches:
        break

print(len(train_dataset_sliced))

# Seed for generating images
seed = tf.random.normal([16, noise_dim])

# Train GAN
train(train_dataset_sliced, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, lambda_gp)


# save Generator loss and Discriminator loss to json
gen_losses = [float(loss) for loss in gen_losses]
disc_losses = [float(loss) for loss in disc_losses]

loss_data = { 
            "Generator_loss" : gen_losses,
            "Discriminator_loss": disc_losses
    }
# write a dictionary to json file
json_path = f'{path}/loss_data.json'

with open(json_path,'w') as json_file : 
     json.dump(loss_data,json_file)
print("Loss data file exported successfully.")
# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(gen_losses, label='Generator Loss')
plt.plot(disc_losses, label='discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses during Training')
plt.savefig(f'{path}/losses_plot.png')
plt.show()
