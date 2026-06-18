---
type: "project" # DON'T TOUCH THIS ! :)
date: "2026-05-29" # Date you first upload your project.
# Title of your project (we like creative title)
title: "Wired for Sound: Functional Connectivity of the Auditory Cortex in Schizophrenia"

# List the names of the collaborators within the [ ]. If alone, simple put your name within []
names: [Garene Matossian, Anna Petroseniak, Yang Jing Zheng, Hira Zahid, Diya Shah, Denisa Lazar]

# Your project GitHub repository URL
github_repo: https://github.com/brainhack-school2026/BHS_2026_Schizophrenia_AVH_Project

# If you are working on a project that has website, indicate the full url including "https://" below or leave it empty.
website:

# List +- 4 keywords that best describe your project within []. Note that the project summary also involves a number of key words. Those are listed on top of the [github repository](https://github.com/PSY6983-2021/project_template), click `manage topics`.
# Please only lowercase letters
tags: [schizophrenia, dysconnectivity, hallucinations, brainhack]

# Summarize your project in < ~75 words. This description will appear at the top of your page and on the list page with other projects..

summary: "This project investigates auditory cortex functional connectivity in schizophrenia using open-source fMRI data from OpenNeuro. Using a seed-to-voxel connectivity approach centered on the left auditory cortex, we examined differences in brain network communication among schizophrenia patients with and without auditory verbal hallucinations (AVH+ and AVH-) and healthy controls during a speech perception task."

# If you want to add a cover image (listpage and image in the right), add it to your directory and indicate the name
# below with the extension.
image: ""
---
<!-- This is an html comment and this won't appear in the rendered page. You are now editing the "content" area, the core of your description. Everything that you can do in markdown is allowed below. We added a couple of comments to guide your through documenting your progress. -->

## Project Definition: Background, Research Question, Project Objectives

### Background

Schizophrenia is a chronic psychiatric disorder that affects approximately 24 million individuals worldwide and is increasingly regarded as a disorder of brain network dysconnectivity. Rather than reflecting dysfunction within a single brain region, symptoms may emerge from altered communication among distributed neural systems.

Auditory verbal hallucinations (AVHs) are among the most common positive symptoms of schizophrenia and affect approximately 60 to 80% of individuals with schizophrenia. Neuroimaging studies suggest that AVHs involve abnormal interactions between auditory processing regions, language networks, and systems involved in self-monitoring and internally generated speech (CITE). In addition, previous work has consistently implicated the left auditory cortex, particularly regions surrounding Heschl's gyrus, in the pathophysiology of hallucinations (CITE).

This project utilizes OpenNeuro dataset ds004302 from Soler-Vidal et al. (2022), which includes fMRI data acquired during a speech perception task from healthy controls, schizophrenia patients without auditory verbal hallucinations (AVH−), and schizophrenia patients experiencing auditory verbal hallucinations (AVH+).

**Main Question**
Do healthy controls, AVH− participants, and AVH+ participants differ in the functional connectivity of the auditory cortex during speech perception?

**Research Objectives**
1. Learn neuroimaging preprocessing and quality control workflows using fMRIPrep.
2. Develop skills in seed-based functional connectivity analysis using Nilearn.
3. Apply reproducible neuroimaging workflows to open-source schizophrenia datasets.
4. Examine auditory cortex connectivity patterns associated with hallucination status.
5. Gain experience using high-performance computing resources and collaborative software development tools.
   
## Methodology: Workflow, Tools Used

**Workflow & Procedures**
1. Data Organization: Raw neuroimaging data were obtained from OpenNeuro (ds004302) and organized according to BIDS standards.

2. fMRI Preprocessing: Functional and anatomical MRI data were preprocessed using fMRIPrep. Processing included motion correction, anatomical-functional coregistration, tissue segmentation, normalization to MNI space, and confound extraction. Slice timing correction was omitted because the primary objective was seed-based functional connectivity analysis. Skull stripping was skipped because the raw neuroimaging data already underwent skull stripping.

3. Seed Definition: A seed region corresponding to the left auditory cortex was identified using the Schaefer 2018 functional atlas. The selected parcel was the cortical region nearest to a literature-based left Heschl's gyrus coordinate (MNI: −42, −26, 10).

4. Functional Connectivity: The average BOLD signal from the auditory cortex seed was extracted for each participant. Connectivity between the seed and every voxel in the brain was calculated using Pearson correlation.

5. Fisher-z Transformation: Correlation coefficients were transformed using Fisher's z transformation to improve suitability for statistical analyses.

6. Statistical Analysis Between Groups: Connectivity maps were compared between healthy controls, AVH− participants, and AVH+ participants using two-sample t-tests and one-way ANOVA.

### Tools Used

In summary, the dataset was obtained from OpenNeuro and organized according to the Brain Imaging Data Structure, or BIDS, which provides a standardized framework for managing neuroimaging data. Computationally intensive analyses were performed on SciNet, a high-performance computing cluster.

For preprocessing, we used fMRIPrep and FreeSurfer, while functional connectivity analyses were conducted in Python using Nilearn. Throughout the project, we used VS Code, Jupyter Notebooks, and the command line interface for coding and analysis, while GitHub was used for version control and project organization to support reproducibility and collaboration. 

**Neuroimaging Software**
1. fMRIPrep (v25.2.4)
2. Nilearn
3. FreeSurfer

**Computing Environment**
1. SciNet Teach Cluster
3. Python/Jupyter Notebooks
4. VS Code/Terminal

**Data Standards and Version Control**
1. OpenNeuro
2. Brain Imaging Data Structure (BIDS)
3. Git and GitHub

<img width="992" height="503" alt="Screenshot 2026-06-17 at 8 01 10 PM" src="https://github.com/user-attachments/assets/2173bffa-0cd3-4a8b-8792-ed60db32e14a" />

## Data

### Data Characteristics

The complete dataset (OpenNeuro, ds004302) contains 71 participants across three groups. Due to time constraints, analyses were initially conducted on a subset of 27 participants consisting of:
*9 Healthy Controls
*9 AVH− Participants
*9 AVH+ Participants

### Deliverables

At the end of this project, we will have:
1. Reproducible GitHub repository
2, README file (project report)
3. fMRI preprocessing script
4. Seed-based functional connectivity
5. Group-level FC script
6. BrainHack presentation

### Repository Contents

The repository contains:
1. Preprocessing scripts
2. Functional connectivity scripts
3. Group-level statistical analysis code
4. Documentation and workflow descriptions (git log, commits)
5. BrainHack presentation materials
6. Figures and outputs

## Results: Skills Learned, Preliminary Findings

### Overview 

A complete preprocessing and functional connectivity workflow was successfully implemented using fMRIPrep and Nilearn. Preprocessing was conducted on the SciNet Teach cluster using containerized workflows. Seed-to-voxel connectivity maps were generated using the left auditory cortex as the seed region and subsequently analyzed at the group level.

### Skills We Learned

**Neuroimaging Analysis**
* BIDS-compliant dataset organization
* fMRIPrep preprocessing workflows
* Quality-control assessment of MRI data
* Seed-based functional connectivity analysis
* Group-level neuroimaging statistics

**Computational Skills**
* Teach cluster usage 
* Containerized neuroimaging workflows using Apptainer
* Git and GitHub version control
* Python-based neuroimaging analyses

### Preliminary Findings [to be updated]

Seed-based connectivity analyses revealed differences in auditory cortex connectivity among healthy controls, AVH− participants, and AVH+ participants. Exploratory group comparisons identified widespread connectivity differences between groups. However, after correction for multiple comparisons, no clusters remained statistically significant, likely reflecting the limited sample size used in this pilot analysis. Nonetheless, these findings demonstrate the feasibility of the workflow and provide a foundation for future analyses using the full dataset.

## Conclusion: Wrap-Up, Future Directions

### Future Directions
1. Complete preprocessing of the full dataset (n = 71)
2. Perform whole-sample connectivity analyses
3. Investigate connectivity within auditory, language, salience, sensorimotor, and default mode networks
4. Examine relationships between connectivity patterns and hallucination status
5. Explore demographic and clinical moderators, including age and sex
