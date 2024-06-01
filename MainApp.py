import streamlit as st
from streamlit_drawable_canvas import st_canvas 
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import gannet_model as net

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def demo(args):
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location="cpu"))
    model.eval()

    st.title("Inpainting Demo")
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 30)
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    
    
    
    if bg_image is not None:
        image = Image.open(bg_image)
        width, height = image.width, image.height
        # Read uploaded image
        orig_img = image
        # st.image(orig_img, caption="Uploaded Image", use_column_width=True)

        # Initialize mask canvas
        mask_canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0.5)",  # Set initial fill color to semi-transparent white
                stroke_width=stroke_width,
            stroke_color="rgb(255, 255, 255)",
            background_image=orig_img,
            height=height,
            width=width,
            update_streamlit = realtime_update,
            drawing_mode="freedraw"
        )

        if mask_canvas.image_data is not None:
            mask_img = Image.fromarray(mask_canvas.image_data.astype("uint8"))
            # st.image(mask_img, caption="Mask", use_column_width=True)

            if st.button("Predict masked region"):
                # Convert PIL image to grayscale
                pil_image_gray = mask_img.convert("L")

                # Convert PIL image to NumPy array
                np_image = np.array(pil_image_gray)

                # Expand dimensions to [h, w, 1]
                np_image = np.expand_dims(np_image, axis=-1)

                # Convert data type to np.uint8
                np_image = np_image.astype(np.uint8)
                # print(np_image)
                mask_array = np.array(np_image)
                mask_tensor = ToTensor()(mask_array).unsqueeze(0)

                img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
                with torch.no_grad():
                    masked_tensor = (img_tensor * (1 - mask_tensor.float())) + mask_tensor
                    pred_tensor = model(masked_tensor, mask_tensor)
                    comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)

                    comp_np = postprocess(comp_tensor[0])
                    st.image(comp_np, caption="Inpainted Image", use_column_width=True)

            if st.button("Reset Mask"):
                mask_canvas.image_data = None

if __name__ == "__main__":
    class Args:
        thick = 15
        pre_train = "Ptrained model/nature/G0000000.pt"
        dir_image = "img"
        dir_mask = "../../dataset"
        data_train = "places2"
        data_test = "places2"
        image_size = 512
        mask_type = "pconv"
        block_num = 8
        rates = [1, 2, 4, 8]
        gan_type = "smgan"
        seed = 2021
        num_workers = 4
        lrg = 1e-4
        lrd = 1e-4
        optimizer = "ADAM"
        beta1 = 0.5
        beta2 = 0.999
        rec_loss = {"L1": 1.0, "Style": 250.0, "Perceptual": 0.1}
        adv_weight = 0.01
        iterations = 1000000
        batch_size = 8
        port = 22334
        resume = False
        print_every = 10
        save_every = 10000
        save_dir = "../experiments"
        tensorboard = False
        painter = "freeform"

    # Load pretrained model
    demo(Args)
