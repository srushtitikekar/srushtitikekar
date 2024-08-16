# GANs: ACGAN, DCGAN, and WGAN Implementation and Evaluation

This repository contains the implementation of three different types of Generative Adversarial Networks (GANs): Auxiliary Classifier GAN (ACGAN), Deep Convolutional GAN (DCGAN), and Wasserstein GAN (WGAN). The project focuses on designing these models and evaluating their performance on image generation tasks.

## Project Description

Generative Adversarial Networks are a class of machine learning frameworks designed to generate new data samples that mimic a given distribution. This project implements and compares three GAN variants to understand their strengths and limitations.

### 1. Auxiliary Classifier GAN (ACGAN)

- **Overview**: ACGAN introduces class labels into the GAN framework, allowing for conditional image generation. The discriminator not only distinguishes between real and fake images but also predicts the class of the images.
- **Implementation**: The generator is conditioned on the class labels, and the discriminator is trained to output both the real/fake decision and the class label. This allows the model to generate images that belong to a specified class.
- **Performance**: The model's performance is evaluated based on the quality of the generated images and the accuracy of the class predictions.

### 2. Deep Convolutional GAN (DCGAN)

- **Overview**: DCGAN is a popular GAN variant that leverages deep convolutional neural networks for both the generator and discriminator. It is particularly effective in generating realistic images.
- **Implementation**: The generator uses transposed convolutional layers to upscale the input noise vector into an image, while the discriminator uses standard convolutional layers to distinguish between real and generated images.
- **Performance**: DCGAN's performance is assessed by the realism of the generated images and the stability of the training process.

### 3. Wasserstein GAN (WGAN)

- **Overview**: WGAN addresses the instability issues of traditional GANs by introducing a new loss function based on the Earth Mover's (Wasserstein) distance. This leads to more stable training and better quality images.
- **Implementation**: The discriminator (referred to as the critic in WGAN) is designed to approximate the Wasserstein distance between the real and generated data distributions. The model uses weight clipping to enforce the Lipschitz constraint required by the Wasserstein distance.
- **Performance**: WGAN is evaluated based on the stability of the training process, the diversity of the generated images, and the quality of the images.

## Evaluation and Comparison

The three GAN models are evaluated on the following criteria:

- **Image Quality**: Assessed visually and quantitatively using metrics like Inception Score (IS) and Fréchet Inception Distance (FID).
- **Training Stability**: Evaluated by monitoring the loss curves and the consistency of the generated images throughout the training process.
- **Model Complexity**: Compared based on the computational resources required and the ease of training.

## Key Takeaways

- **ACGAN**: Best suited for conditional image generation tasks where class labels are available.
- **DCGAN**: Provides a balance between simplicity and performance, ideal for general image generation tasks.
- **WGAN**: Offers the most stable training, making it a good choice for generating high-quality images, especially when dealing with complex datasets.

## Conclusion

This project provides an in-depth analysis of three popular GAN architectures, highlighting their advantages and limitations. The results demonstrate that while each GAN variant has its strengths, the choice of model depends on the specific requirements of the task at hand.


