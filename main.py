import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    if config.train_path is None:
        config.train_path = os.path.join(config.dataset_base_path, 'train/')
    if config.valid_path is None:
        config.valid_path = os.path.join(config.dataset_base_path, 'valid/')
    if config.test_path is None:
        config.test_path = os.path.join(config.dataset_base_path, 'test/')

    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','DCTU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
        
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


# In main.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    # The original image size is 518x392. We use a size divisible by 16 for U-Net compatibility.
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512], help='image size (height, width)')
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3, help='image input channels')
    parser.add_argument('--output_ch', type=int, default=3, help='image output channels')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4) # Increase if your GPU allows
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.5)

    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--val_step', type=int, default=1) # Validate every epoch

    # misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    # --- THIS IS THE NEW BLOCK TO ADD ---
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--cuda_idx', type=int, default=0)

# New argument for the base dataset directory
    parser.add_argument('--dataset_base_path', type=str, default='./dataset/', help='Base directory of the dataset containing train, valid, and test subfolders.')

# Optional arguments for specific paths (these will override the base path if used)
    parser.add_argument('--train_path', type=str, default=None, help='Path to training images. Overrides --dataset_base_path.')
    parser.add_argument('--valid_path', type=str, default=None, help='Path to validation images. Overrides --dataset_base_path.')
    parser.add_argument('--test_path', type=str, default=None, help='Path to testing images. Overrides --dataset_base_path.')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None, help='List of GPU indices to use (e.g., 0 1 2). Uses all available if not specified.')

    config = parser.parse_args()
    
    # Convert image_size list to a tuple
    config.image_size = tuple(config.image_size)
    
    # Remove the random hyperparameter section from the original main function
    main(config)

