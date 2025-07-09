# In solver.py

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image

# Import your models and evaluation metrics
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, DCT_UNet
from evaluation import get_ssim, get_psnr

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.model = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = nn.L1Loss() # L1 Loss for image-to-image regression
        self.augmentation_prob = config.augmentation_prob

        # Hyperparameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Training settings
        self.model_type = config.model_type
        self.t = config.t # for R2U_Net and R2AttU_Net
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
    # In Colab, cuda_idx is not needed as you're assigned one GPU.
    # We will let PyTorch manage the device.
            self.device = torch.device("cuda")
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.build_model(config)

    def build_model(self, config):
        """Create a model and its optimizer."""
        if self.model_type == 'U_Net':
            self.model = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.model = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.model = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.model = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'DCTU_Net':
            self.model = DCT_UNet(img_ch=self.img_ch, output_ch=self.output_ch)

        # --- ADD THIS MULTI-GPU LOGIC ---
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
        # Use the GPU IDs specified by the user, or all available GPUs if None.
            self.model = nn.DataParallel(self.model, device_ids=config.gpu_ids)
    
    # Move the model (or the wrapped model) to the primary device
        self.model.to(self.device)
    # --- END OF NEW LOGIC ---   
        self.optimizer = Adam(list(self.model.parameters()), self.lr, [self.beta1, self.beta2])

    def train(self):
        """Train the model."""
        best_ssim = 0.0
        lr = self.lr

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            for i, (images, gt) in enumerate(self.train_loader):
                images = images.to(self.device)
                gt = gt.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, gt)
                epoch_loss += loss.item()

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                if (i+1) % self.log_step == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

            # Decay learning rate
            if (epoch+1) > self.num_epochs_decay:
                lr -= (self.lr / (self.num_epochs - self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print(f'Decayed learning rate to {lr:.6f}')
            
            # Validation and saving the best model
            if (epoch+1) % self.val_step == 0:
                val_ssim, val_psnr = self.validate()
                print(f'--- Validation | Epoch [{epoch+1}/{self.num_epochs}] ---')
                print(f'SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}')
                
                # Save the model if it has the best SSIM so far
                if val_ssim > best_ssim:
                    best_ssim = val_ssim
                    best_model_path = os.path.join(self.model_path, f'{self.model_type}-best_model.pth')
                    # torch.save(self.model.state_dict(), best_model_path)
                    if isinstance(self.model, nn.DataParallel):
                        torch.save(self.model.module.state_dict(), best_model_path)
                    else:
                        torch.save(self.model.state_dict(), best_model_path)
                    print(f'>>> Model saved to {best_model_path} (SSIM: {best_ssim:.4f})')
                print('-----------------------------------------')


    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_ssim, total_psnr = 0.0, 0.0
        with torch.no_grad():
            for i, (images, gt) in enumerate(self.valid_loader):
                images = images.to(self.device)
                gt = gt.to(self.device)
                outputs = self.model(images)
                
                total_ssim += get_ssim(outputs, gt)
                total_psnr += get_psnr(outputs, gt)
                
        return total_ssim / len(self.valid_loader), total_psnr / len(self.valid_loader)

    def test(self):
        """Test the model and measure inference time."""
        # Load the best saved model
        best_model_path = os.path.join(self.model_path, f'{self.model_type}-best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        total_ssim, total_psnr = 0.0, 0.0
        total_inference_time = 0.0
        num_images = 0

        with torch.no_grad():
            for i, (images, gt) in enumerate(self.test_loader):
                images = images.to(self.device)
                gt = gt.to(self.device)

                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()
                
                total_inference_time += (end_time - start_time)
                num_images += images.size(0)

                total_ssim += get_ssim(outputs, gt)
                total_psnr += get_psnr(outputs, gt)
                
                # Save some sample results
                if i < 10: # save first 10 batches of results
                    result_image = torch.cat([images * 0.5 + 0.5, gt, outputs], dim=0) # Denormalize input
                    save_image(result_image, os.path.join(self.result_path, f'test_{i+1}.png'), nrow=self.batch_size)
        
        avg_ssim = total_ssim / len(self.test_loader)
        avg_psnr = total_psnr / len(self.test_loader)
        avg_inference_time = total_inference_time / num_images
        
        print(f'--- Test Results for {self.model_type} ---')
        print(f'Average SSIM: {avg_ssim:.4f}')
        print(f'Average PSNR: {avg_psnr:.4f}')
        print(f'Average Inference Time per Image: {avg_inference_time * 1000:.2f} ms')
        print('-----------------------------------------')
