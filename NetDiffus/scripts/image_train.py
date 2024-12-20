"""
Train a diffusion model on images.
"""

'''
How to launch it:

 python NetDiffus/scripts/image_train.py --data_dir materiale/dataset/datasetOriginaleInNPZ--image_size 10 --num_channels 128 
 --num_res_blocks 3 --diffusion_steps 100 --noise_schedule cosine --learn_sigma True --class_cond True 
 --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 4
'''

import torch
print(f'{torch.cuda.is_available()=}')

import argparse
import sys
import os

base_dir = os.path.dirname(__file__)

# Costruisci il percorso risalendo di due livelli
target_dir = os.path.abspath(os.path.join(base_dir, '../../'))

# Cambia la directory di lavoro
os.chdir(target_dir)

print("Current working directory:", os.getcwd())

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
import dist_util
import logger
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
        data_dir="/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/dataset/datasetOriginaleInNPZ",  # Directory del dataset
        schedule_sampler="uniform",  # Campionamento uniforme
        lr=4e-4,
        weight_decay=0.0,  # Nessuna penalizzazione dei pesi
        lr_anneal_steps=0,  # Nessun annealing
        batch_size=1,  # Batch più grande per GPU medie
        microbatch=-1,  # Disabilitato
        ema_rate="0.999",  # EMA leggermente più reattivo
        log_interval=30,  # Frequenza dei log invariata
        save_interval=2000,  # Salva il modello ogni 1000
        resume_checkpoint="",  # Default, nessun checkpoint
        use_fp16=True,  # Disabilitato per compatibilità GPU
        fp16_scale_growth=1e-3,  # Non usato con FP16 disabilitato
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
