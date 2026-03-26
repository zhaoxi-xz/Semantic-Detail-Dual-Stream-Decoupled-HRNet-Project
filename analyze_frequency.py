import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ================= 配置区域 =================
# 数据路径 (请根据你的实际路径修改)
data_root = r"E:\desk\Semantic_Segmentation_FCN-master 2\selected_samples_for_fft"
plastic_dir = os.path.join(data_root, "plastic_shed")
solar_dir = os.path.join(data_root, "solar_greenhouse")

# 输出目录
output_dir = r"E:\desk\Semantic_Segmentation_FCN-master 2\frequency_analysis_final"
os.makedirs(output_dir, exist_ok=True)

# 切片大小
PATCH_SIZE = 64


# ===========================================

def extract_inner_patch(image, mask, patch_size=64):
    """从mask内部提取纯净的切片"""
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=4)
    valid_pixels = np.where(eroded_mask > 0)

    if len(valid_pixels[0]) == 0:
        return None

    h, w = mask.shape
    half_size = patch_size // 2

    for _ in range(50):
        idx = random.randint(0, len(valid_pixels[0]) - 1)
        cy, cx = valid_pixels[0][idx], valid_pixels[1][idx]
        y1, y2 = cy - half_size, cy + half_size
        x1, x2 = cx - half_size, cx + half_size

        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            continue

        mask_roi = binary_mask[y1:y2, x1:x2]
        if cv2.countNonZero(mask_roi) == patch_size * patch_size:
            return image[y1:y2, x1:x2]
    return None


def compute_fft_spectrum(image_patch):
    """计算切片的FFT对数幅度谱"""
    gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1e-8)
    return magnitude_spectrum


def compute_laplacian_variance(image_patch):
    """计算切片的拉普拉斯方差"""
    gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def process_samples_patch_mode(sample_dir, sample_type):
    """处理流程：提取切片 -> 计算指标"""
    image_dir = os.path.join(sample_dir, "images")
    mask_dir = os.path.join(sample_dir, "masks")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png', '.jpg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    laplacian_variances = []
    accumulated_fft = None
    count = 0
    valid_patches = []

    for img_file, mask_file in tqdm(zip(image_files, mask_files), desc=f"Processing {sample_type}",
                                    total=len(image_files)):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        patch = extract_inner_patch(image, mask, patch_size=PATCH_SIZE)

        if patch is None:
            continue

        fft = compute_fft_spectrum(patch)
        var = compute_laplacian_variance(patch)

        if accumulated_fft is None:
            accumulated_fft = fft
        else:
            accumulated_fft += fft

        laplacian_variances.append(var)
        count += 1

        if len(valid_patches) < 1:
            valid_patches.append(patch)

    avg_fft = accumulated_fft / count if count > 0 else None
    avg_lap = np.mean(laplacian_variances) if count > 0 else 0

    print(f"[{sample_type}] 成功提取切片数: {count}/{len(image_files)}")
    return avg_fft, avg_lap, valid_patches


def plot_radial_profile(fft_spectrum, title, ax, color):
    """绘制径向分布曲线"""
    center_x, center_y = fft_spectrum.shape[1] // 2, fft_spectrum.shape[0] // 2
    y, x = np.indices(fft_spectrum.shape)
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), fft_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    ax.plot(radialprofile[2:], label=title, color=color, linewidth=2.5)


# ================= 辅助函数：保存单张图片 =================
def save_single_figure(data, save_path, title=None, cmap=None, vmin=None, vmax=None, add_cbar=False, cbar_label=""):
    """
    通用绘图保存函数，用于保存 heatmap 或 image
    """
    plt.figure(figsize=(6, 5))  # 保持一个适中的长宽比

    # 绘图
    if cmap:
        im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = plt.imshow(data)  # 原图不需要cmap

    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=10)

    plt.axis('off')  # 去除坐标轴

    # 添加色标 (如果需要)
    if add_cbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 去除白边保存
    plt.close()
    print(f"已保存: {save_path}")


def main():
    # 锁定随机种子，保证每次运行结果一致 (这对论文配图很重要)
    random.seed(2025)
    np.random.seed(2025)

    # 1. 运行分析
    print("--- 开始内部切片分析 (Patch Size: {}) ---".format(PATCH_SIZE))
    p_fft, p_lap, p_patches = process_samples_patch_mode(plastic_dir, "plastic_shed")
    s_fft, s_lap, s_patches = process_samples_patch_mode(solar_dir, "solar_greenhouse")

    # 2. 打印关键数据
    print("\n" + "=" * 40)
    print(f"塑料拱棚 (高频) 平均拉普拉斯方差: {p_lap:.2f}")
    print(f"日光温室 (低频) 平均拉普拉斯方差: {s_lap:.2f}")
    print("=" * 40 + "\n")

    if p_fft is None or s_fft is None:
        print("错误：数据不足，无法绘图。")
        return

    # ================= 核心修改：分图保存 =================
    print("正在为论文排版生成独立图表...")

    # 定义可视化参数 (保持统一的量纲以便对比)
    VIS_VMIN = 60
    VIS_VMAX = 160
    diff_spectrum = p_fft - s_fft

    # --- 图1: 塑料拱棚切片原图 ---
    if p_patches:
        save_single_figure(
            p_patches[0],
            os.path.join(output_dir, "1_plastic_patch.png"),
            title="Plastic Shed Patch"
        )

    # --- 图2: 日光温室切片原图 ---
    if s_patches:
        save_single_figure(
            s_patches[0],
            os.path.join(output_dir, "2_solar_patch.png"),
            title="Solar Greenhouse Patch"
        )

    # --- 图3: 塑料拱棚平均频谱 (带Colorbar) ---
    save_single_figure(
        p_fft,
        os.path.join(output_dir, "3_plastic_spectrum.png"),
        title="Avg Spectrum (Plastic)",
        cmap='jet', vmin=VIS_VMIN, vmax=VIS_VMAX,
        add_cbar=True, cbar_label="Log Magnitude"
    )

    # --- 图4: 日光温室平均频谱 (带Colorbar) ---
    save_single_figure(
        s_fft,
        os.path.join(output_dir, "4_solar_spectrum.png"),
        title="Avg Spectrum (Solar)",
        cmap='jet', vmin=VIS_VMIN, vmax=VIS_VMAX,
        add_cbar=True, cbar_label="Log Magnitude"
    )

    # --- 图5: 差值热力图 ---
    save_single_figure(
        diff_spectrum,
        os.path.join(output_dir, "5_difference_map.png"),
        title="Difference Map (Plastic - Solar)",
        cmap='bwr', vmin=-15, vmax=15,
        add_cbar=True, cbar_label="Energy Diff"
    )

    # --- 图6: 径向分布曲线 (单独绘制) ---
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    plot_radial_profile(p_fft, "Plastic Shed", ax, '#1f77b4')
    plot_radial_profile(s_fft, "Solar Greenhouse", ax, '#ff7f0e')

    plt.title("Radial Power Spectrum Density", fontsize=16, fontweight='bold', pad=10)
    plt.xlabel("Frequency (Radius)", fontsize=14)
    plt.ylabel("Log Magnitude", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=12)

    radial_path = os.path.join(output_dir, "6_radial_curve.png")
    plt.tight_layout()
    plt.savefig(radial_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {radial_path}")

    print(f"\n所有图片已单独保存至: {output_dir}")
    print("提示：在论文排版时，你可以将这些图片任意组合（例如 2x3 或 3x2 网格）。")


if __name__ == "__main__":
    main()