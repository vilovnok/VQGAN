from types import SimpleNamespace

from .train import TrainVQGAN
from .utils import *



args = SimpleNamespace(
    latent_dim=256,
    image_size=256,
    num_codebook_vectors=1024,
    beta=0.25,
    image_channels=3,
    dataset_path=r"celeba/img_align_celeba/img_align_celeba",
    device="cuda:1",
    batch_size=16,
    epochs=60,
    learning_rate=2.25e-5,
    beta1=0.5,
    beta2=0.9,
    disc_start=10000,
    disc_factor=1.0,
    rec_loss_factor=1.0,
    perceptual_loss_factor=1.0
)

def main():
    dataset_zip = "celeba-dataset.zip"
    extract_folder = "celeba"

    download_celeba(dataset_zip)
    extract_dataset(dataset_zip, extract_folder)
    TrainVQGAN(args)



if __name__ == "__main__":
    main()