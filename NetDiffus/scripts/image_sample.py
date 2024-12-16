'''
python NetDiffus/scripts/image_sample.py 
--model_path NetDiffus/128/iterate/df/synth_models/model055000.pt 
--num_sample 64  --batch_size 16

  python NetDiffus/scripts/image_sample.py  --model_path 128/iterate/df/synth_models/model001000.pt --image_size 10 
  --num_channels 128 --num_res_blocks 3 --diffusion_steps 100 --noise_schedule cosine --learn_sigma True 
  --class_cond True --rescale_learned_sigmas False --rescale_timesteps False

'''



import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import dist_util, logger
from script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from pathlib import Path
base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, "../../materiale/dataset/datasetConNPZ")
output_dir = os.path.join(base_dir,"../../128/iterate/df/synth_models/luigiNpzSint")
# Recupera il percorso delle sottocartelle del dataset
#dataset_dir = Path("/home/franc_ubuntu/Università/Progetto/materiale/dataset/datasetConNPZ")  # Cambia con il percorso del tuo dataset
class_folders = sorted(os.listdir(dataset_dir))  
NUM_CLASSES = len(class_folders)  
#output_dir = "/home/franc_ubuntu/Università/Progetto/NetDiffus/128/iterate/df/synth_models"


# Crea un dizionario che mappa l'indice della classe al nome
class_names = {i: class_folder for i, class_folder in enumerate(class_folders)}

def save_samples_to_npz(samples, class_idx, output_dir):
    """
    Salva i dati generati per ogni classe come un singolo file .npz.
    """
    class_name = class_names[class_idx]
    class_folder = os.path.join(output_dir)
    os.makedirs(class_folder, exist_ok=True)

    # Prepara il percorso del file .npz
    output_path = os.path.join(class_folder, f"{class_name}_synthetic_samples.npz")
    
    # Salva tutte le immagini come file .npz
    np.savez(output_path, images=samples)
    print(f"Saved {output_path} with shape {samples.shape}")

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

    for class_idx, class_folder in enumerate(class_folders):
        # Genera immagini per ogni classe specifica
        model_kwargs = {}
        if args.class_cond:
            classes = th.full((args.num_samples,), class_idx, dtype=th.long, device=dist_util.dev())
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.num_samples, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = sample.permute(0, 2, 3, 1)  # Cambia l'ordine dei canali per salvarli
        sample = sample.contiguous().cpu().numpy()  # Converte a NumPy

        # Salva tutte le immagini generate per questa classe in un unico file .npz
        save_samples_to_npz(sample, class_idx, output_dir="/Users/luigivessella/Desktop/Università/data analitycs/Progetto/128/iterate/df/synth_models/luigiNpzSint")

        logger.log(f"Generated {args.num_samples} samples for class {class_folder}")

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10, #numero di sample generati per classe
        batch_size=10,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
