# Inpainting with GANs - Demo Application

This project implements an image inpainting system using a Generative Adversarial Network (GAN). The demo allows users to mask parts of an image and predict the missing regions using the trained model. It is built with `PyTorch`, `Streamlit`, and a custom GAN architecture for image inpainting.

## Features

- **Image Inpainting**: Recover missing regions from an image using a GAN-based model.
- **Interactive Canvas**: Draw on the image to mask areas that need inpainting.
- **Pretrained Model**: Load a pretrained model for performing inpainting.
- **Streamlit Interface**: An easy-to-use web interface built with Streamlit.

## Project Structure

- **`gannet_model.py`**: Contains the GAN architecture including the generator and discriminator models.
- **`MainApp.py`**: Implements the Streamlit interface for loading images, applying masks, and performing inpainting using the loaded model.
- **`requirements.txt`**: List of required Python libraries for running the project.

## GAN Architecture

The GAN is implemented using a custom generator and discriminator:

- **Generator**: The inpainting model takes an image and a mask as inputs and predicts the missing content using a convolutional neural network (CNN) with up-sampling layers and AOTBlocks.
- **Discriminator**: A convolutional network with spectral normalization is used to distinguish between real and generated (inpainted) images.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/inpainting-gan-demo.git
   cd inpainting-gan-demo
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pretrained model and place it in the appropriate directory (`Ptrained model/nature/G0000000.pt`). You can modify the `Args` class in `MainApp.py` to load a different pretrained model.

## Usage

To run the inpainting demo application, follow these steps:

1. Launch the application:

   ```bash
   streamlit run MainApp.py
   ```

2. Upload an image in `.png` or `.jpg` format.

3. Use the canvas to draw the mask on the image.

4. Click **Predict masked region** to perform inpainting and view the result.

5. You can click **Reset Mask** to reset the mask and try again.

## Example

Below is an example of inpainting using the GAN model. The first image is the original image, the second is the masked image, and the final image shows the inpainted result.

| Original Image | Masked Image | Inpainted Image |
| :------------: | :----------: | :-------------: |
| ![Original](images/original.png) | ![Masked](images/masked.png) | ![Inpainted](images/inpainted.png) |

## Model Architecture

The GAN model consists of:

1. **Generator**:
   - Encoder-decoder structure with skip connections.
   - Utilizes AOTBlock for dilated convolutions.
   - Output is generated using `tanh` activation.

2. **Discriminator**:
   - Uses spectral normalization for stability.
   - Leaky ReLU activations and multiple convolutional layers.

## Customization

- You can adjust the GAN parameters (e.g., block numbers, dilation rates, mask type) by modifying the `Args` class in `MainApp.py`.
- Experiment with different pretrained models by placing them in the appropriate directory and updating the path in the `Args` class.

## Acknowledgements

Special thanks to the PyTorch and Streamlit communities for their excellent tools and libraries.

---
