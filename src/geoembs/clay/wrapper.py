import torch
import loguru
from .module import ClayMAEModule
import numpy as np
from einops import rearrange, reduce, repeat

logger = loguru.logger

means = np.array([53.33853489, 44.41999383, 35.96075039])
stds = np.array([50.44633167, 43.54469652, 44.63162242])


class ClayWrapper:

    def __init__(self, metadata_path, checkpoint_path):

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.checkpoint_path = checkpoint_path
        self.metadata_path = metadata_path

        logger.info(f"using device {self.device}")

        logger.info("creating clay model instance")
        self.clay_model = ClayMAEModule(
            model_size="large",
            mask_ratio=0.75,
            norm_pix_loss=False,
            patch_size=8,
            shuffle=True,
            metadata_path=metadata_path,
            teacher="vit_large_patch14_reg4_dinov2.lvd142m",
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            lr=5e-06,
            wd=0.05,
            b1=0.9,
            b2=0.95,
            embeddings_level="mean",
        ).to(self.device)

        logger.info("loading clay model weights")
        z = torch.load(
            checkpoint_path, weights_only=False, map_location=torch.device(self.device)
        )
        self.clay_model.load_state_dict(z["state_dict"])

        # mean and stds for normalization of RGB channels
        self.means = means
        self.stds = stds

        logger.info("done")

    def batch_embeddings(self, batch):
        """
        batch: [batch_size, 3, img_size, img_size]
               the 3 is three channels for rgb

               the imgs are assumed to be ints in [0,255]
        """

        def print_device_info(model, x):
            # Check model device
            print("\n=== Model Device Info ===")
            print(f"Model parameters are on:", next(model.parameters()).device)

            # Check input tensors
            print("\n=== Input Tensors Device Info ===")
            for key, value in x.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.device}")
                elif isinstance(value, dict):
                    print(f"{key}: (nested dict)")
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: {v.device}")

        def to_device(x, device):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in x.items()
            }

        if not batch.shape[1] == 3:
            raise ValueError(
                f"expecting 3 channels (rgb), but found {batch.shape[1]}"
            )
        image_size = batch.shape[-1]

        batch_normalized = np.transpose(
            (np.transpose(batch, [0, 2, 3, 1]) - self.means) / self.stds, [0, 3, 1, 2]
        )

        x = {
            "pixels": torch.tensor(batch_normalized).type(torch.float),
            "time": torch.zeros([len(batch_normalized), 4]),
            "latlon": torch.zeros([len(batch_normalized), 4]),
            "gsd": torch.tensor(10.0),
            "waves": torch.tensor([1552.0, 1355.0, 1105.0]),
        }  # rgb freqs

        x = to_device(x, next(self.clay_model.parameters()).device)
        # print_device_info(self.clay_model, x)

        with torch.no_grad():
            embeddings_raw, *_ = self.clay_model.model.encoder(x)

        patch_size = self.clay_model.model.patch_size
        # compute patch and image embeddings
        patch_embeddings = rearrange(
            embeddings_raw[:, :-1, :],  # :-1; last embedding is the cls_token
            "b (h w) d -> b h w d",
            w=image_size // patch_size // 2,
            h=image_size // patch_size // 2,
        )
        # image embeddings
        e = reduce(patch_embeddings, "b h w d -> b d", "mean")
        return e
