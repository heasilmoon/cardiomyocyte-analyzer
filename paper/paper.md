---
title: 'Cardiomyocyte Analyzer: a lightweight, open-source tool for quantifying cardiomyocyte contraction, calcium transients, and morphology from video microscopy'
tags:
  - Python
  - cardiomyocyte
  - contraction analysis
  - calcium imaging
  - cell morphology
  - image analysis
  - bioimage informatics
authors:
  - name: "[YOUR FULL NAME]"
    orcid: "[YOUR ORCID, e.g. 0000-0000-0000-0000]"
    affiliation: 1
affiliations:
  - name: "[YOUR INSTITUTION / LAB]"
    index: 1
date: "[SUBMISSION DATE]"
bibliography: paper.bib
---

<!--
JOSS SUBMISSION CHECKLIST (delete this comment block before submitting)
- [ ] Fill in author name(s), ORCID(s), affiliation(s), date above
- [ ] Fill in the Validation section below with real numbers once you've
      run the agreement analysis (see README.md "검증 워크플로우") against
      Fiji/MUSCLEMOTION or another reference method on your lab's data
- [ ] Double-check every citation in paper.bib against the original source
      (author list, year, volume/pages) — they were drafted from general
      knowledge and were not verified against the live publication record
- [ ] Add a few example figures (plots this tool produces) if useful
- [ ] Make sure the GitHub repo has: LICENSE, CITATION.cff, tests that pass
      in CI, and a tagged release matching what you submit
- [ ] Read the current JOSS author guidelines before submitting — format/
      requirements can change: https://joss.readthedocs.io/en/latest/submitting.html
-->

# Summary

Cardiomyocyte Analyzer is an open-source, Python-based web application for
quantifying three of the most common measurements made on cardiomyocyte
video microscopy: contraction (beating) kinetics, calcium transients, and
cell/tissue morphology, in both 2D and 3D. It is designed as a lightweight
alternative to general-purpose platforms such as Fiji/ImageJ
[@schindelin2012fiji] for labs that only need these specific analyses and
want a smaller, easier-to-deploy installation. The tool runs as a single
FastAPI process that serves both the analysis API and a browser-based
interface, requiring no Fiji/Java installation, no plugin management, and no
manual macro scripting.

# Statement of need

Fiji, and plugins built for it such as MUSCLEMOTION [@sala2018musclemotion]
for contraction analysis, are the de facto standard in the cardiomyocyte
research community. They are powerful and well validated, but the full Fiji
distribution is large (several hundred megabytes to over a gigabyte with
plugins), requires a Java runtime, and analyses are typically run
interactively through the GUI or hand-written macros, which makes batch
processing and integration into automated pipelines cumbersome for labs
without dedicated bioimage-analysis support.

Cardiomyocyte Analyzer addresses this gap for the specific, common subset of
analyses most cardiomyocyte labs need day to day:

- **Beating analysis**, following the same pixel-intensity-difference
  principle as MUSCLEMOTION [@sala2018musclemotion], with a reference-frame
  displacement mode and automatic, per-recording tuning of peak-detection
  parameters via autocorrelation of the contraction signal.
- **Calcium transient analysis**: dF/F0 normalization, transient detection,
  rise time, and exponential decay time constant.
- **2D/3D morphology**: watershed-based separation of touching cells, and
  population alignment scoring (a circular order parameter in 2D, a nematic
  order parameter in 3D).
- **Two-group statistical comparison** across an arbitrary number of videos
  per condition (e.g. control vs. treatment), using Mann-Whitney U tests
  appropriate for the small sample sizes typical of this kind of experiment.

The tool is intended for researchers who want fast, scriptable,
batch-friendly analysis of cardiomyocyte video without installing or
maintaining a full Fiji environment.

# Validation

<!-- Fill this in with real numbers from your lab's data. Suggested
structure: report N independent recordings compared against Fiji/
MUSCLEMOTION (or manual/expert annotation) per analysis type, with Pearson/
Spearman correlation, ICC(2,1) absolute-agreement, and Bland-Altman bias +
95% limits of agreement. The `/api/validate/agreement` endpoint in this
repository computes all of these directly from a CSV of paired
(this-tool, reference-method) values and produces a scatter + Bland-Altman
figure suitable for a manuscript. -->

[VALIDATION RESULTS GO HERE — see README.md for the step-by-step workflow
to produce them with this repository's tooling.]

# Acknowledgements

[Acknowledge funding, collaborators, or lab support here, if applicable.]

# References
