"""Exploracion descriptiva del dataset ENSANUT 2024 (adolescentes).
Genera estadisticas PONDERADAS y figuras en /figuras.
Las prevalencias se calculan usando el factor de expansion (ponde_f).
"""

# Standard library
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Local
from utils import (
    AGRESOR_LBL,
    CAT_BLUE,
    CAT_GREEN,
    CAT_LAVENDER,
    CAT_MAUVE,
    CAT_PEACH,
    CAT_RED,
    CAT_SAPPHIRE,
    COLORS,
    METODO_LBL,
)

DATA_PATH: Path = Path("data/adolescentes_limpio.csv")
FIG_DIR: Path = Path("figuras/exploracion")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR: Path = Path("output/estadistica")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    }
)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
W = "ponderador"

SEXOS: list[str] = ["Hombre", "Mujer"]


def wpct(num_mask, denom_mask):
    """Prevalencia ponderada: sum(w | numerador) / sum(w | denominador) * 100."""
    return df.loc[num_mask, W].sum() / df.loc[denom_mask, W].sum() * 100


def wpct_sub(sub_df, num_mask):
    """Prevalencia ponderada sobre un subconjunto ya filtrado."""
    return sub_df.loc[num_mask, W].sum() / sub_df[W].sum() * 100


def add_weighted_note(fig, y=-0.08):
    fig.text(
        0.5,
        y,
        "Prevalencias ponderadas con factor de expansión (ENSANUT 2024)",
        ha="center",
        fontsize=8,
        style="italic",
    )


print("Exploracion ENSANUT 2024 – Adolescentes (PREVALENCIAS PONDERADAS)")

# [0] Calidad del dataset
print("\n[0] CALIDAD DEL DATASET")
print(f"  Filas    : {len(df)}")
print(f"  Columnas : {len(df.columns)}")
dupes = df.duplicated().sum()
print(f"  Duplicados: {dupes}" + (" <- AVISO" if dupes > 0 else ""))
print("\n  Valores perdidos en variables clave:")
for col in ["asi", "ideacion", "intento", "sexo", "edad", W]:
    n = df[col].isna().sum()
    flag = " <- AVISO" if n > 0 else ""
    print(f"    {col:<15}: {n} ({n / len(df) * 100:.1f}%){flag}")

# [1] Distribución por sexo (ponderada)
total = len(df)
w_total = df[W].sum()
w_hombres = df.loc[df["sexo"] == "Hombre", W].sum()
w_mujeres = df.loc[df["sexo"] == "Mujer", W].sum()
pct_h = w_hombres / w_total * 100
pct_m = w_mujeres / w_total * 100

print(f"\n[1] TOTAL ENTREVISTADOS: {total} (población expandida: {w_total:,.0f})")
print(f"    Hombres : n={(df['sexo'] == 'Hombre').sum():>5}  ({pct_h:.1f}% ponderado)")
print(f"    Mujeres : n={(df['sexo'] == 'Mujer').sum():>5}  ({pct_m:.1f}% ponderado)")

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    [w_hombres, w_mujeres],
    labels=["Hombres", "Mujeres"],
    colors=[COLORS["Hombre"], COLORS["Mujer"]],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax.set_title(f"Distribución por sexo\n(n = {total})")
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "01_distribucion_sexo.png")
plt.close(fig)
print("    -> fig: 01_distribucion_sexo.png")

# [2] Edad (media ponderada)
edad_media_w = np.average(df["edad"], weights=df[W])
print(f"\n[2] EDAD  media_ponderada={edad_media_w:.1f}  min={df['edad'].min()}  max={df['edad'].max()}")

fig, ax = plt.subplots(figsize=(10, 4))
bin_edges = np.arange(10, 21)
bar_w = 0.4
for i, sexo in enumerate(SEXOS):
    sub = df[df["sexo"] == sexo]["edad"]
    counts, _ = np.histogram(sub, bins=bin_edges)
    centers = bin_edges[:-1] + 0.5
    ax.bar(
        centers + (i - 0.5) * bar_w, counts, width=bar_w, label=sexo, color=COLORS[sexo], edgecolor="white", alpha=0.85
    )
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Frecuencia (n)")
ax.set_title("Distribución de edad por sexo")
ax.legend()
ax.set_xticks([x + 0.5 for x in range(10, 20)])
ax.set_xticklabels(range(10, 20))
fig.tight_layout()
fig.savefig(FIG_DIR / "02_distribucion_edad.png")
plt.close(fig)
print("    -> fig: 02_distribucion_edad.png")

# [3] Prevalencia de ASI (ponderada)
asi_valid = df["asi"].notna()
asi_denom = asi_valid.sum()
asi_n = (df["asi"] == 1).sum()
asi_pct = wpct(df["asi"] == 1, asi_valid)

print(f"\n[3] ASI PREVALENCIA PONDERADA (denominador respondentes: n={asi_denom})")
print(f"    Total : n={asi_n} ({asi_pct:.1f}% ponderado)")

rows = []
for sexo in SEXOS:
    mask_sex = df["sexo"] == sexo
    mask_num = mask_sex & (df["asi"] == 1)
    mask_den = mask_sex & asi_valid
    n = mask_num.sum()
    pct = df.loc[mask_num, W].sum() / df.loc[mask_den, W].sum() * 100
    rows.append({"Sexo": sexo, "n_ASI": n, "n_total": mask_den.sum(), "pct": pct})
    print(f"    {sexo:<8}: n={n:>4} / {mask_den.sum()} ({pct:.1f}% ponderado)")

fig, ax = plt.subplots(figsize=(6, 4))
labels = [r["Sexo"] for r in rows] + ["Total"]
pcts = [r["pct"] for r in rows] + [asi_pct]
ns = [r["n_ASI"] for r in rows] + [asi_n]
colores = [COLORS[lbl] for lbl in labels[:-1]] + [COLORS["Total"]]
bars = ax.bar(labels, pcts, color=colores, edgecolor="white")
for bar, n in zip(bars, ns, strict=False):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"n={n}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("% con ASI (ponderado)")
ax.set_title("Prevalencia de ASI por sexo\n(ponderada)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, max(pcts) * 1.3)
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "03_prevalencia_asi_sexo.png")
plt.close(fig)
print("    -> fig: 03_prevalencia_asi_sexo.png")

# [4] Edad al ASI (ponderada entre víctimas)
asi_df = df[df["asi"] == 1].copy()
asi_edad_val = asi_df["abuso_sexual"].notna()
base_edad = asi_df[asi_edad_val]
den_edad = len(base_edad)

w_antes = base_edad.loc[base_edad["abuso_sexual"] == 1, W].sum()
w_despues = base_edad.loc[base_edad["abuso_sexual"] == 2, W].sum()
w_edad_total = w_antes + w_despues
pct_antes = w_antes / w_edad_total * 100
pct_despues = w_despues / w_edad_total * 100

print(f"\n[4] EDAD AL ASI (sobre {den_edad} casos con dato válido, % ponderado)")
print(f"    Antes de 12 años : n={(base_edad['abuso_sexual'] == 1).sum()} ({pct_antes:.1f}%)")
print(f"    12 años o más    : n={(base_edad['abuso_sexual'] == 2).sum()} ({pct_despues:.1f}%)")

data_edad = {}
for sexo in SEXOS:
    sub = base_edad[base_edad["sexo"] == sexo]
    wa = sub.loc[sub["abuso_sexual"] == 1, W].sum()
    wd = sub.loc[sub["abuso_sexual"] == 2, W].sum()
    data_edad[sexo] = {"Antes de 12": wa, "12 o más": wd, "n": len(sub)}

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, (sexo, counts) in zip(axes[:2], data_edad.items(), strict=False):
    vals = [counts["Antes de 12"], counts["12 o más"]]
    ax.pie(
        vals,
        labels=["Antes de 12", "12 o más"],
        autopct="%1.1f%%",
        colors=[CAT_RED, CAT_BLUE],
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        startangle=90,
    )
    ax.set_title(f"{sexo}\n(n={counts['n']})")
axes[2].pie(
    [w_antes, w_despues],
    labels=["Antes de 12", "12 o más"],
    autopct="%1.1f%%",
    colors=[CAT_RED, CAT_BLUE],
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    startangle=90,
)
axes[2].set_title(f"Total\n(n={den_edad})")
fig.suptitle("Edad al momento del ASI (ponderada)", fontweight="bold")
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "04_asi_edad_inicio.png")
plt.close(fig)
print("    -> fig: 04_asi_edad_inicio.png")

# [5] Agresor en ASI (frecuencias ponderadas)
print("\n[5] AGRESOR EN ASI (parentesco_agresor, % ponderado)")
asi_df_ag = asi_df[asi_df["parentesco_agresor"].notna()].copy()
asi_df_ag["agresor_lbl"] = asi_df_ag["parentesco_agresor"].map(AGRESOR_LBL)
ag_w = asi_df_ag.groupby("agresor_lbl")[W].sum().sort_values(ascending=False)
ag_pct = (ag_w / ag_w.sum() * 100).round(1)
ag_n = asi_df_ag["agresor_lbl"].value_counts()
for lbl in ag_w.index:
    print(f"    {lbl:<18}: n={ag_n.get(lbl, 0):<4} ({ag_pct[lbl]:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ag_n_sorted = ag_n.reindex(ag_w.index)
ag_n_sorted.sort_values().plot(kind="barh", ax=axes[0], color=CAT_LAVENDER, edgecolor="white")
axes[0].set_title("Relación agresor-víctima en ASI\n(Total)")
axes[0].set_xlabel("n")
axes[0].set_ylabel("")

ag_sex = {}
for sexo in SEXOS:
    sub = asi_df[(asi_df["sexo"] == sexo) & asi_df["parentesco_agresor"].notna()]
    sub_lbl = sub["parentesco_agresor"].map(AGRESOR_LBL).dropna()
    ag_sex[sexo] = sub_lbl.value_counts()

ag_sex_df = pd.DataFrame(ag_sex).fillna(0)
ag_sex_df.plot(kind="barh", ax=axes[1], color=[COLORS["Hombre"], COLORS["Mujer"]], edgecolor="white")
axes[1].set_title("Relación agresor-víctima por sexo\ndel entrevistado")
axes[1].set_xlabel("n")
axes[1].legend(title="Sexo víctima")
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "05_asi_agresor_relacion.png")
plt.close(fig)
print("    -> fig: 05_asi_agresor_relacion.png")

# [6] Sexo del agresor (ponderado entre víctimas)
print("\n[6] SEXO DEL AGRESOR (ponderado)")
ag_sexo_lbl = {1: "Hombre", 2: "Mujer"}
asi_ag_valid = asi_df[asi_df["sexo_agresor"].notna()].copy()
asi_ag_valid["agresor_sexo_lbl"] = asi_ag_valid["sexo_agresor"].map(ag_sexo_lbl)
ag_sexo_w = asi_ag_valid.groupby("agresor_sexo_lbl")[W].sum()
ag_sexo_n = asi_ag_valid["agresor_sexo_lbl"].value_counts()
ag_sexo_total_w = ag_sexo_w.sum()
for k in ag_sexo_w.index:
    pct = ag_sexo_w[k] / ag_sexo_total_w * 100
    print(f"    {k}: n={ag_sexo_n[k]} ({pct:.1f}% ponderado)")

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    ag_sexo_w.values,
    labels=ag_sexo_w.index,
    colors=[COLORS.get(k, CAT_GREEN) for k in ag_sexo_w.index],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax.set_title("Sexo del agresor en ASI\n(ponderado)")
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "06_asi_sexo_agresor.png")
plt.close(fig)
print("    -> fig: 06_asi_sexo_agresor.png")

# [7] Ideación suicida (ponderada)
ide_valid = df["ideacion"].notna()
ide_n = (df["ideacion"] == 1).sum()
ide_denom = ide_valid.sum()
ide_pct = wpct(df["ideacion"] == 1, ide_valid)
print(f"\n[7] IDEACION SUICIDA PONDERADA (denominador respondentes: n={ide_denom})")
print(f"    Total: n={ide_n} ({ide_pct:.1f}% ponderado)")

rows_ide = []
for sexo in SEXOS:
    mask_sex = df["sexo"] == sexo
    mask_num = mask_sex & (df["ideacion"] == 1)
    mask_den = mask_sex & ide_valid
    n = mask_num.sum()
    pct = df.loc[mask_num, W].sum() / df.loc[mask_den, W].sum() * 100
    rows_ide.append({"Sexo": sexo, "n": n, "pct": pct})
    print(f"    {sexo:<8}: n={n:>4} ({pct:.1f}% ponderado)")

fig, ax = plt.subplots(figsize=(6, 4))
labels_ide = [r["Sexo"] for r in rows_ide] + ["Total"]
pcts_ide = [r["pct"] for r in rows_ide] + [ide_pct]
ns_ide = [r["n"] for r in rows_ide] + [ide_n]
colores_ide = [COLORS[lbl] for lbl in labels_ide[:-1]] + [COLORS["Total"]]
bars = ax.bar(labels_ide, pcts_ide, color=colores_ide, edgecolor="white")
for bar, n in zip(bars, ns_ide, strict=False):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"n={n}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("% con ideación suicida (ponderado)")
ax.set_title("Ideación suicida por sexo\n(ponderada)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, max(pcts_ide) * 1.3)
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "07_ideacion_suicida.png")
plt.close(fig)
print("    -> fig: 07_ideacion_suicida.png")

# [8] Intentos de suicidio (ponderados)
int_n = (df["intento"] == 1).sum()
una_vez = (df["intento_suicidio"] == 1).sum()
dos_mas = (df["intento_suicidio"] == 2).sum()
int_denom = df["intento"].notna().sum()
int_pct = wpct(df["intento"] == 1, df["intento"].notna())
print(f"\n[8] INTENTOS DE SUICIDIO PONDERADOS (denominador respondentes: n={int_denom})")
print(f"    Total: n={int_n} ({int_pct:.1f}% ponderado)")
print(f"    Una vez    : {una_vez}")
print(f"    2 o más    : {dos_mas}")

rows_int = []
for sexo in SEXOS:
    mask_sex = df["sexo"] == sexo
    mask_num = mask_sex & (df["intento"] == 1)
    mask_den = mask_sex & df["intento"].notna()
    n = mask_num.sum()
    pct = df.loc[mask_num, W].sum() / df.loc[mask_den, W].sum() * 100
    rows_int.append(
        {
            "Sexo": sexo,
            "n": n,
            "pct": pct,
            "1vez": (df.loc[mask_sex, "intento_suicidio"] == 1).sum(),
            "2mas": (df.loc[mask_sex, "intento_suicidio"] == 2).sum(),
        }
    )
    print(f"    {sexo:<8}: n={n:>4} ({pct:.1f}% ponderado)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
labels_int = [r["Sexo"] for r in rows_int] + ["Total"]
pcts_int = [r["pct"] for r in rows_int] + [int_pct]
ns_int = [r["n"] for r in rows_int] + [int_n]
bars = axes[0].bar(labels_int, pcts_int, color=[COLORS.get(lbl, COLORS["Total"]) for lbl in labels_int], edgecolor="white")
for bar, n in zip(bars, ns_int, strict=False):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"n={n}", ha="center", va="bottom", fontsize=10
    )
axes[0].set_ylabel("% con intento de suicidio (ponderado)")
axes[0].set_title("Intentos de suicidio por sexo\n(ponderada)")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].set_ylim(0, max(pcts_int) * 1.3)

x = np.arange(len(SEXOS))
w_bar = 0.35
axes[1].bar(x - w_bar / 2, [r["1vez"] for r in rows_int], w_bar, label="1 vez", color=CAT_BLUE, edgecolor="white")
axes[1].bar(
    x + w_bar / 2, [r["2mas"] for r in rows_int], w_bar, label="2 o más veces", color=CAT_PEACH, edgecolor="white"
)
axes[1].set_xticks(x)
axes[1].set_xticklabels([r["Sexo"] for r in rows_int])
axes[1].set_ylabel("n")
axes[1].set_title("Frecuencia de intentos por sexo")
axes[1].legend()
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "08_intentos_suicidio_sexo.png")
plt.close(fig)
print("    -> fig: 08_intentos_suicidio_sexo.png")

# [9] Métodos de intento (frecuencias, n)
int_df = df[df["intento"] == 1].copy()
print(f"\n[9] METODOS DE INTENTO (sobre {len(int_df)} casos)")

method_cols = [c for c in df.columns if c.startswith("metodo_")]
met_counts = {METODO_LBL[c]: int_df[c].sum() for c in method_cols if c in METODO_LBL}
met_series = pd.Series(met_counts).sort_values(ascending=False)
met_series = met_series[met_series > 0]
print(met_series.to_string())

met_sex = {
    sexo: {METODO_LBL[c]: int_df[int_df["sexo"] == sexo][c].sum() for c in method_cols if c in METODO_LBL}
    for sexo in SEXOS
}
met_sex_df = pd.DataFrame(met_sex).loc[met_series.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
met_series.sort_values().plot(kind="barh", ax=axes[0], color=CAT_PEACH, edgecolor="white")
axes[0].set_title("Métodos de intento de suicidio\n(Total)")
axes[0].set_xlabel("n")
met_sex_df.sort_values("Hombre").plot(
    kind="barh", ax=axes[1], color=[COLORS["Hombre"], COLORS["Mujer"]], edgecolor="white"
)
axes[1].set_title("Métodos por sexo")
axes[1].set_xlabel("n")
axes[1].legend(title="Sexo")
fig.tight_layout()
fig.savefig(FIG_DIR / "09_metodos_suicidio.png")
plt.close(fig)
print("    -> fig: 09_metodos_suicidio.png")

# [10] Co-ocurrencia ASI + intento (ponderada)
print("\n[10] ASI e INTENTO DE SUICIDIO (co-ocurrencia, ponderada)")
ambas = df[df["asi"].notna() & df["intento"].notna()]
asi_pos = ambas[ambas["asi"] == 1]
asi_neg = ambas[ambas["asi"] == 0]

pct_asi_w = wpct_sub(asi_pos, asi_pos["intento"] == 1)
pct_noasi_w = wpct_sub(asi_neg, asi_neg["intento"] == 1)
n_asi_int = (asi_pos["intento"] == 1).sum()
n_noasi_int = (asi_neg["intento"] == 1).sum()

print(f"     Base: {len(ambas)} respondentes en ambas variables")
print(f"     Con ASI (n={len(asi_pos)}) -> intentaron: n={n_asi_int} ({pct_asi_w:.1f}% ponderado)")
print(f"     Sin ASI (n={len(asi_neg)}) -> intentaron: n={n_noasi_int} ({pct_noasi_w:.1f}% ponderado)")

fig, ax = plt.subplots(figsize=(6, 4))
groups = ["Con ASI", "Sin ASI"]
pcts_cooc = [pct_asi_w, pct_noasi_w]
ns_cooc = [len(asi_pos), len(asi_neg)]
bars = ax.bar(groups, pcts_cooc, color=[CAT_RED, CAT_BLUE], edgecolor="white")
for bar, pct, n_grp in zip(bars, pcts_cooc, ns_cooc, strict=False):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{pct:.1f}%\n(n={n_grp})",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_ylabel("% que intentó suicidio (ponderado)")
ax.set_title("Intento de suicidio según antecedente de ASI\n(ponderada)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, max(pcts_cooc) * 1.35)
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "10_asi_vs_intento.png")
plt.close(fig)
print("    -> fig: 10_asi_vs_intento.png")

# [11] Violencia general (ponderada)
viol_valid = df["violencia_reciente"].notna()
viol_n = (df["violencia_reciente"] == 1).sum()
viol_denom = viol_valid.sum()
viol_pct = wpct(df["violencia_reciente"] == 1, viol_valid)
print(f"\n[11] VIOLENCIA GENERAL ULTIMOS 12 MESES PONDERADA (n={viol_denom})")
print(f"     Total: n={viol_n} ({viol_pct:.1f}% ponderado)")

rows_viol = []
for sexo in SEXOS:
    mask_sex = df["sexo"] == sexo
    mask_num = mask_sex & (df["violencia_reciente"] == 1)
    mask_den = mask_sex & viol_valid
    n = mask_num.sum()
    pct = df.loc[mask_num, W].sum() / df.loc[mask_den, W].sum() * 100
    rows_viol.append({"Sexo": sexo, "n": n, "pct": pct})
    print(f"     {sexo:<8}: n={n:>4} ({pct:.1f}% ponderado)")

fig, ax = plt.subplots(figsize=(6, 4))
labels_v = [r["Sexo"] for r in rows_viol] + ["Total"]
pcts_v = [r["pct"] for r in rows_viol] + [viol_pct]
ns_v = [r["n"] for r in rows_viol] + [viol_n]
bars = ax.bar(labels_v, pcts_v, color=[COLORS.get(lbl, COLORS["Total"]) for lbl in labels_v], edgecolor="white")
for bar, n in zip(bars, ns_v, strict=False):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"n={n}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("% con violencia (ponderado)")
ax.set_title("Violencia/agresión en últimos 12 meses por sexo\n(ponderada)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, max(pcts_v) * 1.3)
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "11_violencia_general.png")
plt.close(fig)
print("     -> fig: 11_violencia_general.png")

# [12] Denuncia del ASI (ponderada entre víctimas)
print("\n[12] DENUNCIA DEL ASI (ponderada)")
den_valid = asi_df[asi_df["denuncia_asi"].notna()]
w_den_si = den_valid.loc[den_valid["denuncia_asi"] == 1, W].sum()
w_den_no = den_valid.loc[den_valid["denuncia_asi"] == 2, W].sum()
w_den_total = w_den_si + w_den_no
pct_den_si = w_den_si / w_den_total * 100
pct_den_no = w_den_no / w_den_total * 100
n_den_si = (den_valid["denuncia_asi"] == 1).sum()
n_den_no = (den_valid["denuncia_asi"] == 2).sum()
print(f"     Sí denunció: n={n_den_si} ({pct_den_si:.1f}% ponderado)")
print(f"     No denunció: n={n_den_no} ({pct_den_no:.1f}% ponderado)")

razon_lbl = {1: "Miedo", 2: "Vergüenza", 3: "Amenazas", 4: "No sabía\nque podía"}
razon = asi_df[asi_df["denuncia_asi"] == 2]["razon_no_denuncia"].map(razon_lbl).dropna().value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
denuncia_lbl = {1: "Sí denunció", 2: "No denunció"}
den_counts = pd.Series({"Sí denunció": n_den_si, "No denunció": n_den_no})
den_counts.plot(kind="bar", ax=axes[0], color=[CAT_GREEN, CAT_RED], edgecolor="white", rot=0)
axes[0].set_title("¿Denunciaron el ASI?")
axes[0].set_ylabel("n")
for bar in axes[0].patches:
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        int(bar.get_height()),
        ha="center",
        va="bottom",
        fontsize=10,
    )
razon.sort_values().plot(kind="barh", ax=axes[1], color=CAT_MAUVE, edgecolor="white")
axes[1].set_title("Razón por la que no denunció")
axes[1].set_xlabel("n")
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "12_denuncia_asi.png")
plt.close(fig)
print("     -> fig: 12_denuncia_asi.png")

# [13] Atención tras el ASI (ponderada entre víctimas)
print("\n[13] ATENCION TRAS EL ASI (ponderada)")
atencion_lbl = {
    1: "Nadie lo atendió",
    2: "Remedios/automedicación",
    3: "Curandero/hierbero",
    4: "Huesero/sobador",
    5: "Encargado comunidad",
    6: "Psicólogo/terapeuta",
    7: "Médico/consultorio",
    8: "Clínica/hospital",
}
aten_valid = asi_df[asi_df["atencion_tras_asi"].notna()].copy()
aten_valid["aten_lbl"] = aten_valid["atencion_tras_asi"].map(atencion_lbl)
aten_w = aten_valid.groupby("aten_lbl")[W].sum().sort_values(ascending=False)
aten_pct = (aten_w / aten_w.sum() * 100).round(1)
aten_n = aten_valid["aten_lbl"].value_counts()
for lbl in aten_w.index:
    print(f"     {lbl:<25}: n={aten_n.get(lbl, 0):<4} ({aten_pct[lbl]:.1f}%)")

# Figure uses n for bars (since these are small counts)
aten_n_sorted = aten_n.reindex(aten_w.index).dropna()
fig, ax = plt.subplots(figsize=(8, 4))
aten_n_sorted.sort_values().plot(kind="barh", ax=ax, color=CAT_SAPPHIRE, edgecolor="white")
ax.set_title("Atención recibida después del ASI")
ax.set_xlabel("n")
fig.tight_layout()
add_weighted_note(fig)
fig.savefig(FIG_DIR / "13_atencion_tras_asi.png")
plt.close(fig)
print("     -> fig: 13_atencion_tras_asi.png")

# Ideación en víctimas de ASI vs no (ponderada) — para el artículo
print("\n[14] IDEACIÓN SUICIDA EN VÍCTIMAS DE ASI (ponderada)")
ide_ambas = df[df["asi"].notna() & df["ideacion"].notna()]
ide_asi_pos = ide_ambas[ide_ambas["asi"] == 1]
ide_asi_neg = ide_ambas[ide_ambas["asi"] == 0]
pct_ide_asi = wpct_sub(ide_asi_pos, ide_asi_pos["ideacion"] == 1)
pct_ide_noasi = wpct_sub(ide_asi_neg, ide_asi_neg["ideacion"] == 1)
print(f"     Con ASI: {pct_ide_asi:.1f}% ponderado")
print(f"     Sin ASI: {pct_ide_noasi:.1f}% ponderado")

# No atención ponderada (para artículo: "X% no recibió atención")
pct_no_atencion = aten_pct.get("Nadie lo atendió", 0)
print(f"\n[15] % SIN ATENCION TRAS ASI (ponderado): {pct_no_atencion:.1f}%")

print("\nExploración completada.")

# Exportar tabla resumen (ponderada)
resumen = pd.DataFrame(
    {
        "Variable": ["ASI", "Ideación suicida", "Intento de suicidio", "Violencia reciente"],
        "Prevalencia_ponderada (%)": [asi_pct, ide_pct, int_pct, viol_pct],
        "n_positivos": [asi_n, ide_n, int_n, viol_n],
        "Denominador (n)": [asi_denom, ide_denom, int_denom, viol_denom],
    }
)
resumen.to_csv(OUT_DIR / "prevalencias_crudas.csv", index=False)
print(f"    -> CSV: {OUT_DIR / 'prevalencias_crudas.csv'}")
