import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle


# ==========================================================
# BEER–LAMBERT FORWARD PROJECTION
# ==========================================================
def beer_lambert_projection(volume, axis=0, mu_scale=2.0):
    mu = volume * mu_scale
    attenuation = np.sum(mu, axis=axis)
    proj = np.exp(-attenuation)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    return proj


# ==========================================================
# EDGE-PRESERVING 3D SHARPENING
# ==========================================================
def unsharp_mask_3d(volume, sigma=1.0, amount=0.6):
    smooth = gaussian_filter(volume, sigma=sigma)
    sharpened = volume + amount * (volume - smooth)
    return np.clip(sharpened, 0, 1)


# ==========================================================
# PHYSICS-BASED CBCT RECONSTRUCTOR
# ==========================================================
class PhysicsCBCTReconstructor:

    def __init__(self, cbct_dir, output_size=(256, 256, 256)):
        self.templates = []
        self.output_size = output_size

        print("📂 Loading CBCT templates...")
        files = sorted(list(Path(cbct_dir).glob("*.nii")) +
                       list(Path(cbct_dir).glob("*.nii.gz")))

        if len(files) == 0:
            raise RuntimeError("❌ No CBCT templates found")

        for f in files:
            vol = nib.load(f).get_fdata().astype(np.float32)
            vol = np.clip(vol, -1000, 3000)
            vol = (vol + 1000) / 4000.0

            t = torch.from_numpy(vol)[None, None]
            t = F.interpolate(
                t, size=output_size,
                mode="trilinear", align_corners=False
            )

            self.templates.append(t.squeeze().numpy())
            print(f"  ✓ {f.name}")

        print(f"✅ Loaded {len(self.templates)} templates\n")

    # ------------------------------------------------------
    def select_best_template(self, input_px):
        losses = []

        for idx, tpl in enumerate(self.templates):
            proj = beer_lambert_projection(tpl)
            proj_px = F.interpolate(
                torch.from_numpy(proj)[None, None],
                size=input_px.shape,
                mode="bilinear", align_corners=False
            ).squeeze().numpy()

            loss = np.mean((input_px - proj_px) ** 2)
            losses.append(loss)
            print(f"  Template {idx+1} MSE: {loss:.6f}")

        best = int(np.argmin(losses))
        print(f"→ Best template: {best+1}\n")
        return self.templates[best].copy()

    # ------------------------------------------------------
    def iterative_refinement(self, volume, input_px, iterations=8):
        print("🔁 Iterative refinement (sharp & PX-safe)")

        D, H, W = volume.shape

        # Depth prior (prevents PX copying)
        z = np.linspace(-1, 1, D)
        depth_kernel = np.exp(-z**2 / 0.25)
        depth_kernel /= depth_kernel.sum()

        for i in range(iterations):

            proj = beer_lambert_projection(volume)
            proj_px = F.interpolate(
                torch.from_numpy(proj)[None, None],
                size=input_px.shape,
                mode="bilinear", align_corners=False
            ).squeeze().numpy()

            error_px = input_px - proj_px
            mean_err = np.mean(np.abs(error_px))

            if mean_err < 1e-4:
                print("  ✓ Early convergence")
                break

            error_hw = F.interpolate(
                torch.from_numpy(error_px)[None, None],
                size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze().numpy()

            # Depth-weighted backprojection
            error_3d = error_hw[None, :, :] * depth_kernel[:, None, None]

            # Physics-consistent multiplicative update
            volume *= np.exp(0.02 * error_3d)
            volume = np.clip(volume, 0, 1)

            # Smooth ONLY early iterations
            if i < iterations // 2:
                volume = gaussian_filter(volume, sigma=0.4)

            print(f"  Iter {i+1}/{iterations} | PX error: {mean_err:.6f}")

        return volume

    # ------------------------------------------------------
    def reconstruct(self, input_px):
        base = self.select_best_template(input_px)
        refined = self.iterative_refinement(base, input_px)

        # Light edge-preserving denoising
        refined = denoise_tv_chambolle(refined, weight=0.015)

        # Final sharpening
        refined = unsharp_mask_3d(refined, sigma=1.0, amount=0.6)

        return refined


# ==========================================================
# INPUT / OUTPUT
# ==========================================================
def load_px(path, size=(512, 1024)):
    img = Image.open(path).convert("L")
    img = img.resize((size[1], size[0]), Image.LANCZOS)
    px = np.array(img, dtype=np.float32) / 255.0
    px = (px - px.min()) / (px.max() - px.min() + 1e-8)
    return px


def visualize(px, cbct, outdir):
    d, h, w = cbct.shape

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(px, cmap="gray")
    axs[0, 0].set_title("Input Panoramic X-ray")

    axs[0, 1].imshow(cbct[d//2], cmap="gray")
    axs[0, 1].set_title("Axial")

    axs[0, 2].imshow(cbct[:, h//2, :], cmap="gray")
    axs[0, 2].set_title("Sagittal")

    axs[1, 0].imshow(cbct[:, :, w//2], cmap="gray")
    axs[1, 0].set_title("Coronal")

    axs[1, 1].imshow(np.max(cbct, axis=0), cmap="gray")
    axs[1, 1].set_title("Axial MIP")

    axs[1, 2].imshow(beer_lambert_projection(cbct), cmap="gray")
    axs[1, 2].set_title("Simulated PX")

    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{outdir}/physics_reconstruction.png", dpi=150)
    plt.close()


# ==========================================================
# MAIN
# ==========================================================
def main():
    PX_PATH = "xray.jpeg"
    CBCT_DIR = "data"
    OUT_DIR = "results"

    Path(OUT_DIR).mkdir(exist_ok=True)

    print("\n🚀 PX → CBCT Reconstruction (Final Stable Version)\n")

    px = load_px(PX_PATH)
    reconstructor = PhysicsCBCTReconstructor(CBCT_DIR)
    cbct = reconstructor.reconstruct(px)

    affine = np.diag([0.4, 0.4, 0.4, 1])
    nib.save(
        nib.Nifti1Image(cbct.astype(np.float32), affine),
        f"{OUT_DIR}/cbct_physics.nii.gz"
    )

    visualize(px, cbct, OUT_DIR)

    print("\n✅ Reconstruction complete")
    print(f"📂 Output saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
