# DCGAN Cartoon Face Image Generator
By Curtis Eng, Gabriel Sahlin, Elias Curl, Austin Foltz


This project was to build and implement a Deep Convolutional Generative Adversarial Network (DCGAN) to generate cartoon face images. The network is trained on a dataset of cartoon faces, and learns to produce new cartoon faces from random noise vectors.

The DCGAN consists of two neural networks:

Generator (G): Takes a random noise vector and generates a fake image.

Discriminator (D): Classifies images as real (from the dataset) or fake (from the Generator).

These networks are trained to "compete" with each out. The Generator tries to trick the Discriminator, while the Discriminator tries to correctly identify real and fake images.

### Dataset
The dataset itself was trained on the images, such as the set below, and each image was processed with a resizing to be 256x256 instead of 500x500, center cropped, and normalized to pixel values of [-1,1].
![Figure_1 1](https://github.com/user-attachments/assets/1a725693-8a5b-446e-bd66-3e663a22b4a6)

### Parameters Used
The key parameters we ended up using was : 
image size : 256
batch size : 64
Latent vector size (nz) : 64
Generator feature size (ngf) 64
Discriminator feature size (ndf) : 64
Number of epochs = 35
Learning Rate (lr) = 0.0002 for the generator, 0.00005 for the discriminator
Adam hyperparameter (beta1) = 0.5

One major note is that our learning rate for the generator is 0.002 and for the discriminator is 0.00005.
These values are important because it allows for the generator to catch up to the discriminator and it shows of the spike in the beginning of our loss graphs. Our parameters for this GAN in the Conv2d is (in, out, kernel = 4, stride = 2, padding = 1) This is because we want the discriminator to condence or generator to expand the images for the GAN.

### Architecture
We used a combination of 7 convolutional transposed layers along with batchNorms, LeakyReLU activations, and tanh to upsample from 3 x 1 x 1 to 3 x 256 x 256 in our generator and 7 convolutional layers with batchNorms, LeakyReLU, and sigmoid to downsample from 3 x 256 x 256 in our discriminator.

### Results
This is the outcome of testing with 64x64 iamge size:
![64Image](https://github.com/user-attachments/assets/10ecdcbd-dc70-4262-bd9c-3ad4b352d9a6)

As the images were reshaped from 500x500 to 64x64, features that had higher variance, such as the hair styles, gave the GAN issues as it couldn't quite tell each style apart from another and resulted in these "Ghost Hair"s. However, it did recognize more consistent patterns such as glasses/sunglasses and (in this training batch) there was a lot of blonde hair present. One way we attempted to correct this was by increasing the size of the images trained.

This is the outcome of testing with 128x128 image size:

![Screenshot 2025-05-13 224516](https://github.com/user-attachments/assets/9ea988a0-85d0-4897-a3ac-ac2a52af692f)

Where the images appear to improve quite a bit by doing this, but with there consistently still being issues with "ghost hairs", we decided to attempt and increase the image size once more.

This is an example output with 256x256 image size:
![Screenshot 2025-05-13 223846](https://github.com/user-attachments/assets/8ddfff10-feff-44a5-9b8f-009a73d69766)

This is the loss during our 256x256 image size:
![Screenshot 2025-05-13 223552](https://github.com/user-attachments/assets/ffa6478a-9bdc-4a63-8683-6c472f24ba42)

Which overall, although it took the longest to train, over each epoch using 256x256 images nearly got rid of all "ghost hairs" with the cost of having widly different colors in the hairstyles and skin colors at times.

### Future improvements
With more time, it would be interesting to see how using a Wasserstein GAN (WGAN) would handle this task. 

### Sources
Dataset : Cartoon Faces : https://www.kaggle.com/datasets/brendanartley/cartoon-faces-googles-cartoon-set?select=cartoonset100k_jpg

Content Resources:

https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

https://medium.com/@dnyaneshwalwadkar/generative-adversarial-network-gan-simplegan-dcgan-wgan-progan-c92389a3c454 ​

https://developers.google.com/machine-learning/gan/discriminator ​

https://medium.com/@danushidk507/understanding-dcgans-deep-convolutional-generative-adversarial-networks-1984bc028bf8 ​

https://en.wikipedia.org/wiki/Generative_adversarial_network#:~:text=A%20generative%20adversarial%20network%20(GAN,his%20colleagues%20in%20June%202014.​

https://medium.com/@sanjithkumar986/inductive-bias-in-deep-learning-1-17a7c3f35381

