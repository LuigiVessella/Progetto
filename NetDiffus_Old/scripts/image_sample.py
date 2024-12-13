import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from pathlib import Path
from PIL import Image

import dist_util, logger
from script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# Recupera il percorso delle sottocartelle del dataset
dataset_dir = Path("/home/franc_ubuntu/Università/Progetto/materiale/dataset/interoDatasetGASF_RGB")  # Cambia con il percorso del tuo dataset
class_folders = sorted(os.listdir(dataset_dir))  
NUM_CLASSES = len(class_folders)  

# Crea un dizionario che mappa l'indice della classe al nome
class_names = {i: class_folder for i, class_folder in enumerate(class_folders)}


def save_sample_images(sample, class_idx, batch_size, output_dir):
    """
    Salva le immagini generate in una cartella specifica per ciascuna classe.
    """
    class_name = class_names[class_idx]  # Ottieni il nome della classe usando l'indice
    # Crea la cartella per la classe se non esiste
    class_folder = os.path.join(output_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # Converti le immagini in formato PIL e salvale con il nome richiesto
    for i in range(batch_size):
        img = sample[i].cpu().numpy()
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        
        # Nome del file per l'immagine
        img_name = f"{class_name}_sintetic_{i}.png"
        output_path = os.path.join(class_folder, img_name)
        img_pil.save(output_path)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")

    all_images = []
    all_labels = []
    for class_idx, class_folder in enumerate(class_folders):
        # Genera immagini per ogni classe specifica
        model_kwargs = {}
        if args.class_cond:
            # Usa l'indice della classe corrente
            classes = th.full((args.batch_size,), class_idx, dtype=th.long, device=dist_util.dev())
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        # Salva le immagini per la classe corrente
        save_sample_images(sample, class_idx, args.batch_size, output_dir="output_samples")

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        logger.log(f"created {len(all_images) * args.batch_size} samples for class {class_folder}")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1400,
        batch_size=100,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
