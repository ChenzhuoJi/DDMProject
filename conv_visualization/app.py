import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from PIL import Image


def generate_sample(preset, size, seed=42):
    np.random.seed(seed)
    if preset == 'checkerboard':
        x = np.zeros((size, size))
        x[::2, ::2] = 1
        x[1::2, 1::2] = 1
    elif preset == 'cross':
        x = np.zeros((size, size))
        mid = size // 2
        w = max(1, size // 8)
        x[mid-w:mid+w+1, :] = 1
        x[:, mid-w:mid+w+1] = 1
    elif preset == 'gradient':
        x = np.tile(np.linspace(-1, 1, size), (size, 1))
    elif preset == 'circle':
        yy, xx = np.ogrid[:size, :size]
        c = size // 2
        r = size // 3
        x = ((xx - c)**2 + (yy - c)**2 <= r**2).astype(float)
    elif preset == 'noise':
        x = np.random.randn(size, size)
    elif preset == 'edge':
        x = np.zeros((size, size))
        x[:, :size//2] = 1
    else:
        x = np.eye(size)
    return x.astype(np.float32)


def compute_output_size(L_in, K, P, D, S):
    return (L_in + 2 * P - D * (K - 1) - 1) // S + 1


def visualize(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, preset, input_size):
    K = kernel_size
    S = stride
    P = padding
    D = dilation
    G = groups

    inp_per_group = in_channels // G
    out_per_group = out_channels // G

    # ── build input ─────────────────────────────────────────
    if in_channels == 1:
        base = generate_sample(preset, input_size)
        inp_np = base[np.newaxis, :, :]
    else:
        channels = []
        for c in range(in_channels):
            base = generate_sample(preset, input_size, seed=42 + c)
            channels.append(base * (0.3 + 0.7 * (c / max(1, in_channels - 1))))
        inp_np = np.stack(channels, axis=0)

    inp_t = torch.from_numpy(inp_np).float().unsqueeze(0)

    # ── forward ─────────────────────────────────────────────
    conv = torch.nn.Conv2d(in_channels, out_channels, K,
                           stride=S, padding=P, dilation=D,
                           groups=G, bias=False)
    with torch.no_grad():
        out_t = conv(inp_t)

    w = conv.weight.data
    out_np = out_t[0].numpy()
    h_out, w_out = out_np.shape[1:]

    # ── figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor='white', constrained_layout=True)
    gs = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.30)

    # ── 1. Input ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if in_channels == 1:
        ax1.imshow(inp_np[0], cmap='viridis')
        ax1.set_title(f"Input\n1×{input_size}×{input_size}", fontsize=10)
    elif in_channels == 3:
        rgb = inp_np.transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        ax1.imshow(rgb)
        ax1.set_title(f"Input (RGB)\n3×{input_size}×{input_size}", fontsize=10)
    else:
        ax1.imshow(inp_np[0], cmap='viridis')
        ax1.set_title(f"Input (ch0 of {in_channels})\n{in_channels}×{input_size}×{input_size}", fontsize=10)
    ax1.axis('off')

    # ── 2. Kernel weights ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:3])
    w_np = w.numpy()
    vmin, vmax = w_np.min(), w_np.max()
    vlim = max(abs(vmin), abs(vmax)) or 1.0

    grid_rows = out_channels
    grid_cols = inp_per_group
    grid_img = np.zeros((grid_rows * K, grid_cols * K))
    for i in range(grid_rows):
        for j in range(grid_cols):
            grid_img[i*K:(i+1)*K, j*K:(j+1)*K] = w_np[i, j]

    im = ax2.imshow(grid_img, cmap='coolwarm', vmin=-vlim, vmax=vlim)
    plt.colorbar(im, ax=ax2, shrink=0.8)

    if G > 1:
        for g in range(G):
            g_r0 = g * out_per_group * K
            g_c0 = g * inp_per_group * K
            rect = Rectangle(
                (g_c0 - 0.5, g_r0 - 0.5),
                inp_per_group * K, out_per_group * K,
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
            )
            ax2.add_patch(rect)
            ax2.text(g_c0 + inp_per_group * K / 2 - 0.5,
                     (g + 1) * out_per_group * K + 0.5,
                     f'Group {g+1}', ha='center', va='top',
                     fontsize=8, color='lime', fontweight='bold')

    ax2.set_title(f"Kernel Weights{'  (groups=' + str(G) + ')' if G > 1 else ''}\n"
                  f"{out_channels}×{inp_per_group}×{K}×{K}", fontsize=10)
    ax2.set_ylabel('Output Channel')
    ax2.set_xlabel('Input Channel (per group)' if G > 1 else 'Input Channel')

    row_ticks = np.arange(K // 2, grid_rows * K, K)
    col_ticks = np.arange(K // 2, grid_cols * K, K)
    ax2.set_yticks(row_ticks)
    ax2.set_yticklabels(range(grid_rows))
    ax2.set_xticks(col_ticks)
    ax2.set_xticklabels(range(grid_cols))
    ax2.tick_params(length=0)

    # ── 3. Output feature maps ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    n = out_channels
    grid_n = int(np.ceil(np.sqrt(n)))
    out_grid = np.zeros((grid_n * h_out, grid_n * w_out))
    for i in range(n):
        r, c = i // grid_n, i % grid_n
        ch = out_np[i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch = (ch - ch_min) / (ch_max - ch_min)
        out_grid[r*h_out:(r+1)*h_out, c*w_out:(c+1)*w_out] = ch
    ax3.imshow(out_grid, cmap='viridis')
    ax3.set_title(f"Output Feature Maps\n{n}×{h_out}×{w_out}", fontsize=10)
    ax3.axis('off')

    # ── 4. Config + formula ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    h_out_exp = compute_output_size(input_size, K, P, D, S)
    info = (
        f"  Configuration\n"
        f"{'─'*26}\n"
        f"  in_channels  = {in_channels}\n"
        f"  out_channels = {out_channels}\n"
        f"  kernel_size  = {K}\n"
        f"  stride       = {S}\n"
        f"  padding      = {P}\n"
        f"  dilation     = {D}\n"
        f"  groups       = {G}\n"
        f"{'─'*26}\n"
        f"  Input:   {in_channels}×{input_size}×{input_size}\n"
        f"  Output:  {out_channels}×{h_out_exp}×{h_out_exp}\n"
        f"  Params:  {w.numel()}\n"
        f"{'─'*26}\n"
        f"  H_out = ⌊(H_in + 2P − D·(K−1) − 1) / S⌋ + 1\n"
        f"       = ⌊({input_size} + {2*P} − {D}·({K}−1) − 1) / {S}⌋ + 1\n"
        f"       = {h_out_exp}"
    )
    ax4.text(0, 0.95, info, fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))

    # ── 5. Receptive field ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    vis_input = inp_np[0] if in_channels > 0 else np.zeros((input_size, input_size))
    ax5.imshow(vis_input, cmap='gray')

    eff_k = (K - 1) * D + 1
    center = input_size // 2
    half = eff_k // 2

    rect = Rectangle(
        (center - half - 0.5, center - half - 0.5),
        eff_k, eff_k, linewidth=2, edgecolor='red', facecolor='none'
    )
    ax5.add_patch(rect)

    for di in range(K):
        for dj in range(K):
            px = center - half + di * D
            py = center - half + dj * D
            cell = Rectangle(
                (px - 0.5, py - 0.5), 1, 1,
                linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.35
            )
            ax5.add_patch(cell)

    ax5.set_title(
        f"Receptive Field (center)\n"
        f"Effective: {eff_k}×{eff_k}  (dilation={D})",
        fontsize=9
    )
    ax5.axis('off')

    # ── 6. Groups explanation or size effect ──────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    if G > 1:
        g_text = (
            f"Grouped Convolution (groups={G})\n"
            f"{'─'*26}\n"
            f"Input split into {G} groups,\n"
            f"each {inp_per_group} channels.\n\n"
            f"Output split into {G} groups,\n"
            f"each {out_per_group} channels.\n\n"
            "Each group only connects\n"
            "within itself.\n\n"
            f"Params per output ch:\n"
            f"  {inp_per_group}×{K}×{K} = {inp_per_group*K*K}\n"
            f"Total params:\n"
            f"  {out_channels}×{inp_per_group}×{K}×{K}\n"
            f"  = {w.numel()}\n\n"
            f"(w/o groups would be\n"
            f" {out_channels}×{in_channels}×{K}×{K}\n"
            f" = {out_channels*in_channels*K*K})"
        )
    else:
        g_text = (
            "Standard Convolution (groups=1)\n"
            f"{'─'*26}\n"
            "Each output channel connects\n"
            "to ALL input channels.\n\n"
            f"Params per output ch:\n"
            f"  {in_channels}×{K}×{K} = {in_channels*K*K}\n\n"
            f"Total params:\n"
            f"  {out_channels}×{in_channels}×{K}×{K}\n"
            f"  = {w.numel()}"
        )
    ax6.text(0, 0.95, g_text, fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))

    # ── 7. Size effect ────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis('off')
    direction = (
        'Same size'
        if input_size == h_out_exp else
        'Downsampled'
        if h_out_exp < input_size else
        'Upsampled'
    )
    comp_text = (
        f"  Size Effect\n"
        f"{'─'*26}\n"
        f"  Input:  {input_size}×{input_size}\n"
        f"  Output: {h_out_exp}×{h_out_exp}\n"
        f"  Status: {direction}\n"
        f"{'─'*26}\n"
        f"  Effect of increasing:\n"
        f"  • kernel_size  → ↓\n"
        f"  • stride       → ↓\n"
        f"  • padding      → ↑\n"
        f"  • dilation     → ↓\n"
        f"                  (↑ receptive field)\n"
        f"  • groups       → no change\n"
        f"                  (↓ params)"
    )
    ax7.text(0, 0.95, comp_text, fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ── Gradio App ──────────────────────────────────────────────────

def update_groups(in_ch, out_ch):
    valid = [g for g in range(1, min(in_ch, out_ch) + 1)
             if in_ch % g == 0 and out_ch % g == 0]
    return gr.update(choices=valid, value=1)


with gr.Blocks(title="Conv2d Visualizer") as demo:
    gr.Markdown(
        "# PyTorch Conv2d 可视化\n"
        "调节下方参数，实时观察卷积核、输出特征图的变化。\n"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            in_channels = gr.Slider(1, 8, 1, step=1, label="in_channels")
            out_channels = gr.Slider(1, 8, 2, step=1, label="out_channels")
            kernel_size = gr.Slider(1, 7, 3, step=1, label="kernel_size")
            stride = gr.Slider(1, 4, 1, step=1, label="stride")
            padding = gr.Slider(0, 4, 0, step=1, label="padding")
            dilation = gr.Slider(1, 4, 1, step=1, label="dilation")
            groups = gr.Dropdown([1, 2], value=1, label="groups")
            preset = gr.Dropdown(
                ["checkerboard", "cross", "gradient", "circle", "noise", "edge", "diagonal"],
                value="cross", label="Input Pattern"
            )
            input_size = gr.Slider(8, 32, 16, step=1, label="Input Size")
            btn = gr.Button("Update", variant="primary", size="lg")

        with gr.Column(scale=2):
            output_img = gr.Image(label="Visualization", type="pil", height=720)

    all_inputs = [in_channels, out_channels, kernel_size,
                  stride, padding, dilation, groups, preset, input_size]

    btn.click(fn=visualize, inputs=all_inputs, outputs=output_img)

    for inp in all_inputs:
        inp.change(fn=visualize, inputs=all_inputs, outputs=output_img)

    in_channels.change(fn=update_groups, inputs=[in_channels, out_channels],
                       outputs=groups)
    out_channels.change(fn=update_groups, inputs=[in_channels, out_channels],
                        outputs=groups)


if __name__ == '__main__':
    demo.launch(server_port=7860, theme=gr.themes.Soft())
