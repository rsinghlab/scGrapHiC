# scGrapHiC: Deep learning-based graph deconvolution for Hi-C using single cell gene expression

Single-cell Hi-C (scHi-C) protocol helps identify cell-type-specific chromatin interactions and sheds light on cell differentiation and disease progression. Despite providing crucial insights, scHi-C data is often underutilized due the high cost and the complexity of the experimental protocol. We present a deep learning framework, \modelname, that predicts pseudo-bulk scHi-C contact maps using pseudo-bulk scRNA-seq data. Specifically, \modelname performs graph deconvolution to extract genome-wide single-cell interactions from a bulk Hi-C contact map using scRNA-seq as a guiding signal. Our evaluations show that \modelname, trained on 7 cell-type co-assay datasets, outperforms typical sequence encoder approaches. For example, \modelname achieves a substantial improvement of $23.2\%$ in recovering cell-type-specific Topologically Associating Domains over the baselines. It also generalizes to unseen embryo and brain tissue samples. \modelname is a novel method to generate cell-type-specific scHi-C contact maps using widely available genomic signals that enables the study of cell-type-specific chromatin interactions.

[*Preprint*]() is available.

![scNODE model overview]()
