# Counterfactual Explanations for Cautious Random Forests (CRF-CF)

This repository relates to the paper **“Counterfactual Explanations for Cautious Random Forests”**, which studies how to generate counterfactual explanations for **indeterminate (set-valued)** predictions produced by **Cautious Random Forests (CRF)**. 

## What problem does the paper address?

In high-stakes settings, a classifier may prefer to output a **set of possible classes** (e.g., `{c1, c2}`) when the prediction is uncertain, instead of committing to a single label. Such models are called **cautious classifiers**. :contentReference[oaicite:1]{index=1}

A **Cautious Random Forest (CRF)** extends classical random forests by combining the **Imprecise Dirichlet Model** with **belief functions** to produce **indeterminate predictions** under epistemic or aleatoric uncertainty (often near decision boundaries). 

The paper proposes using **counterfactual examples** to explain:
- **why** a prediction is indeterminate, and
- **how** to minimally change feature values so the output becomes **determinate** (a single class). 

## Core idea

Generating counterfactuals for CRF is framed as an optimization problem: find a **valid** counterfactual that is
- **close** to the query instance (proximity),
- **sparse** (few feature changes),
- **plausible** (not an outlier),
- and yields a **determinate** prediction. 

To solve this efficiently, the paper introduces an exact **branch-and-bound** search tailored to random-forest decision regions (box-shaped regions from axis-aligned splits), with:
- **Filtering** to shrink the search space early, 
- **Actionability constraints** (feature restrictions) integrated into the search, 
- **Plausibility checking** using **Local Outlier Factor (LOF)** (k=20, threshold 1.5) to reject outlier-like counterfactuals. 
- **Feature-importance–guided ordering** (global and local, e.g., MDI/PFI/SHAP-FI/SHAP/LIME) to accelerate search without degrading quality metrics. 

## Experimental summary

- Evaluated on **10 UCI binary tabular datasets**
- Compared against **MO, DisCERN, OFCC, LIRE** baselines (chosen for guaranteed validity and practical efficiency). 
- Main findings:
  - The proposed method improves **proximity** and **sparsity** compared to baselines, while maintaining strong **plausibility** (often around or above 90% under LOF<1.5). 
  - Runtime is higher than the fastest baselines, but still typically efficient. 
  - Using **global feature importance** usually yields the most consistent speed-ups across datasets. 

<!-- ## Reference

If you use this work, please cite the paper: xxx -->
