import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Cosmological parameters
# -----------------------------
H0_planck = 67.4   # km/s/Mpc (CMB-inferred H0 under ΛCDM)
H0_shoes  = 73.0   # km/s/Mpc (local SH0ES)
sigma_H0_planck = 0.5
sigma_H0_shoes  = 1.0

Om = 0.315  # matter density parameter
Ol = 1.0 - Om

# -----------------------------
# CPL background: w(z) = w0 + wa z/(1+z)
# -----------------------------
def Ez_cpl(z, w0, wa):
    """Dimensionless H(z)/H0 for a CPL dark-energy model."""
    z = np.asarray(z)
    Om_loc = Om
    Ol_loc = 1.0 - Om_loc
    f = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om_loc * (1 + z) ** 3 + Ol_loc * f)

# ΛCDM: w0 = -1, wa = 0
def H_lcdm(z):
    return H0_planck * Ez_cpl(z, w0=-1.0, wa=0.0)

# Siamese toy model (v4): very close to ΛCDM at z ≳ 1.5,
# with a mild late-time deviation that raises H0.
def H_siamese(z):
    w0_s = -0.97   # slightly less negative than -1
    wa_s = -0.25   # mild evolution, mainly at low z
    H0_s = H0_shoes
    return H0_s * Ez_cpl(z, w0=w0_s, wa=wa_s)

# -----------------------------
# Redshift grid
# -----------------------------
z = np.linspace(0, 3.0, 400)

# -----------------------------
# Schematic observational H(z) points
# (chosen to lie roughly on ΛCDM for neutrality)
# -----------------------------
z_data = np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.3])
H_data = H_lcdm(z_data)                  # centred on ΛCDM
H_err  = np.array([5.0, 5.0, 6.0, 7.0, 9.0, 12.0])  # illustrative uncertainties

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 5))

# 1) Theoretical curves
plt.plot(z, H_lcdm(z), label=r"$\Lambda$CDM", linewidth=2)
plt.plot(z, H_siamese(z), label="Siamese cosmology", linestyle="--", linewidth=2)

# 2) Schematic H(z) points
plt.errorbar(
    z_data,
    H_data,
    yerr=H_err,
    fmt="o",
    ms=5,
    label=r"Schematic $H(z)$ data (illustrative only)",
    alpha=0.85,
)

# 3) H0 bands (z ≈ 0 constraints)
plt.axhspan(
    H0_planck - sigma_H0_planck,
    H0_planck + sigma_H0_planck,
    alpha=0.15,
    label=r"$H_0$ Planck (CMB-inferred)",
)
plt.axhspan(
    H0_shoes - sigma_H0_shoes,
    H0_shoes + sigma_H0_shoes,
    alpha=0.15,
    label=r"$H_0$ SH0ES (local)",
)

plt.xlabel("Redshift $z$")
plt.ylabel(r"$H(z)\, [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
plt.title(r"Hubble expansion: $\Lambda$CDM vs Siamese cosmology")

plt.xlim(0, 3.0)
plt.ylim(60, 240)

# Legend in desired order
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1, 2, 3, 4]  # ΛCDM, Siamese, data, Planck, SH0ES
plt.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="upper left",
    fontsize=8,
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("Hz_comparison_siamese_vs_lcdm_v4_final.png", dpi=300)
plt.show()
