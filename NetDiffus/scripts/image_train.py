"""
Train a diffusion model on images.
"""

import argparse
import os
import torch
print(torch.__version__)  # Verifica la versione di PyTorch
print(torch.cuda.is_available())  # Verifica se CUDA è disponibile

#os.chdir('/share/home/snir5742/torch-env/guided-diffusion-main')
name_dir = os.path.dirname(__file__)

# Costruisci il percorso risalendo di due livelli
base_dir = os.path.abspath(os.path.join(name_dir, '../../../'))

target_dir=os.path.join(base_dir, "guided-diffusion")

# Cambia la directory di lavoro
os.chdir(target_dir)

print("Current working directory:", os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import guided-difusion-0.0.0 as guided_diffusion
#import guided_diffusion
#from guided_diffusion import dist_util, logger
#from guided_diffusion.image_datasets import load_data
#from guided_diffusion.resample import create_named_schedule_sampler
#from guided_diffusion.script_util import (
#    model_and_diffusion_defaults,
#    create_model_and_diffusion,
#    args_to_dict,
#    add_dict_to_argparser,
#)
#from guided_diffusion.train_util import TrainLoop
import dist_util, logger
from image_datasets import load_data
from resample import create_named_schedule_sampler
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from train_util import TrainLoop


def main():
    # Seleziona dispositivo
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Usa la GPU NVIDIA
        print("Using NVIDIA GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")  # Usa la CPU se nessuna GPU è disponibile
        print("Using CPU")

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
