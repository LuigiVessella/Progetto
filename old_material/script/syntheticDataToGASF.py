import numpy as np
import os
from PIL import Image


def save_images_from_npz(npz_file_path, output_directory, label_names):
    # Load the .npz file
    data = np.load(npz_file_path)

    # Check for the image array and the labels
    if 'arr_0' not in data:
        raise KeyError("The .npz file does not contain 'arr_0'.")
    if 'arr_1' not in data:
        raise KeyError("The .npz file does not contain 'arr_1' (labels).")

    images = data['arr_0']  # The images array
    labels = data['arr_1']  # The labels array

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through images and labels
    for i, (img, label) in enumerate(zip(images, labels)):
        # Convert image to RGBA
        img_normalized = (img / 255).clip(0, 1)  # Normalize pixel values
        img_pil = Image.fromarray((img_normalized * 255).astype('uint8'), mode='RGB')

        # Get the class name from the label
        label_name = label_names[int(label)]  # Convert label to class name

        # Create the class folder if it doesn't exist
        label_folder = os.path.join(output_directory, label_name)
        os.makedirs(label_folder, exist_ok=True)

        # Generate the image filename
        image_filename = f"{label_name}_sintetic_{i+1}.png"  # Start numbering from 1

        # Save the image in the corresponding class folder
        output_path = os.path.join(label_folder, image_filename)
        img_pil.save(output_path)

    print(f"Saved {len(images)} images to {output_directory}.")



# Define label names mapping
label_names = [
    "ClashRoyale", "Crunchyroll", "Discord", "JitsiMeet", "KakaoTalk",
    "Line", "Meets", "Omlet", "Signal", "Slack", "Telegram", "Trueconf", "Twitch", "Whatsapp"
]


base_dir = os.path.dirname(__file__)
npz_file_path = os.path.join(base_dir,'../../128/iterate/df/synth_models/samples_1400x10x10x3.npz')  # Update with actual path
output_directory = os.path.join(base_dir,'../dataset/1400ImmaginiSinteticheModello30')  # Update with desired output folder

save_images_from_npz(npz_file_path, output_directory, label_names)
