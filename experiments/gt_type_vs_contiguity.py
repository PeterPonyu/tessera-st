from pathlib import Path
import json
from scipy import stats
import numpy as np

d = json.load(open(str(Path(__file__).resolve().parent / 'expanded_bench.json')))
rows = d['rows']

# map platform string (as it appears in expanded_bench.json) -> Table 1 GT type category
gt_type = {
    'DLPFC(Visium,layer)': 'layer',
    'seqFISH(embryo,ctype)': 'cell type',
    'MERFISH(hypo,domain)': 'domain',
    'STARmap(cortex,region)': 'region',
    'osmFISH(cortex,Region)': 'region',
    'MIBI-TOF(protein,Cluster)': 'cluster',
    'BRCA(Visium,tumor)': 'tumour',
    'IMC(breast,ctype)': 'cell type',
    'openST(HNSCC,annot)': 'annotation',
    'SlideseqV2(hippo,Region)': 'region',
    'CODEX(spleen,niche)': 'niche',
}

structural = {'domain', 'region', 'layer', 'tumour'}
identity = {'cell type', 'niche', 'annotation', 'cluster'}

contig = []
adv = []
binary = []
labels = []
for r in rows:
    p = r['platform']
    t = gt_type[p]
    b = 1 if t in structural else 0
    contig.append(r['GT_contiguity'])
    adv.append(r['spatial_advantage'])
    binary.append(b)
    labels.append((p, t, b, r['GT_contiguity'], r['spatial_advantage']))

contig = np.array(contig)
adv = np.array(adv)
binary = np.array(binary)

rho_c, p_c = stats.spearmanr(contig, adv)
rho_b, p_b = stats.spearmanr(binary, adv)
# point-biserial (pearson between binary and continuous) as alt
pb_r, pb_p = stats.pointbiserialr(binary, adv)

print("n =", len(rows))
print("Spearman contiguity vs advantage: rho=%.3f p=%.4f" % (rho_c, p_c))
print("Spearman binary GT-type vs advantage: rho=%.3f p=%.4f" % (rho_b, p_b))
print("Point-biserial binary GT-type vs advantage: r=%.3f p=%.4f" % (pb_r, pb_p))
print()
for l in sorted(labels, key=lambda x: -x[3]):
    print(l)

# misclassification check: which structural-labeled datasets have low contiguity (would look like identity)
print()
print("Datasets where categorical label disagrees with continuous contiguity ranking (structural but low contiguity, or identity but high contiguity):")
med = np.median(contig)
for p,t,b,c,a in labels:
    predicted_high = (b==1)
    actual_high = (c >= med)
    if predicted_high != actual_high:
        print(" MISMATCH:", p, t, "contiguity=", c, "binary=", b)
