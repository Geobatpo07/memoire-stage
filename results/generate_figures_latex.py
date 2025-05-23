import os
from collections import defaultdict

def group_by_prefix_suffix(files):
    grouped = defaultdict(lambda: defaultdict(list))
    for f in files:
        if not f.endswith(".pdf") or "_" not in f:
            grouped["autres"]["autres"].append(f)
            continue
        parts = f[:-4].split("_")
        prefix = parts[0]      # e.g. 'delta'
        suffix = "_".join(parts[2:]) if len(parts) > 2 else parts[-1]  # e.g. 'euler'
        grouped[prefix][suffix].append(f)
    return grouped

def write_single_figure_block(f, label=None, caption=None):
    block = []
    block.append("\\begin{figure}[h!]")
    block.append("    \\centering")
    block.append(f"    \\includegraphics[width=0.8\\textwidth]{{{f}}}")
    if caption:
        block.append(f"    \\caption{{{caption}}}")
    if label:
        block.append(f"    \\label{{fig:{label}}}")
    block.append("\\end{figure}\n")
    return "\n".join(block)

def write_subfigure_block(figs, suffix, prefix):
    block = []
    block.append("\\begin{figure}[h!]")
    block.append("    \\centering")
    for f in sorted(figs):
        label = f.replace(".pdf", "").replace(" ", "_")
        short = label.split("_")[1]  # 'a', 'b', 'theta', 'E'
        block.append(f"    \\begin{{subfigure}}[b]{{0.45\\textwidth}}")
        block.append(f"        \\includegraphics[width=\\linewidth]{{{f}}}")
        block.append(f"        \\caption{{{short}}}")
        block.append(f"        \\label{{fig:{label}}}")
        block.append("    \\end{subfigure}")
    block.append(f"    \\caption{{{prefix.capitalize()} pour la méthode {suffix}}}")
    block.append(f"    \\label{{fig:{prefix}_{suffix}}}")
    block.append("\\end{figure}\n")
    return "\n".join(block)

def generate_latex_figures_subfigure(output_file="figures_generated.tex"):
    files = sorted(f for f in os.listdir() if f.endswith(".pdf"))
    grouped = group_by_prefix_suffix(files)

    with open(output_file, "w", encoding="utf-8") as f:
        for prefix in grouped:
            f.write(f"% --- Catégorie : {prefix.upper()} ---\n")
            f.write(f"\\section*{{{prefix.capitalize()}}}\n\n")
            for suffix in grouped[prefix]:
                figs = grouped[prefix][suffix]
                if len(figs) >= 2:
                    f.write(write_subfigure_block(figs, suffix, prefix))
                else:
                    single_fig = figs[0]
                    label = single_fig.replace(".pdf", "")
                    caption = label.replace("_", " ").capitalize()
                    f.write(write_single_figure_block(single_fig, label=label, caption=caption))
                f.write("\n")

    print(f"Fichier LaTeX avec subfigures généré : {output_file}")

if __name__ == "__main__":
    generate_latex_figures_subfigure()
