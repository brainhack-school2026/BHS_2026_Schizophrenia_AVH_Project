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

summary: "Our project focuses on investigating the functional connectivity of the auditory cortex with regions implicated in auditory verbal hallucination in individuals with schizophrenia."

# If you want to add a cover image (listpage and image in the right), add it to your directory and indicate the name
# below with the extension.
image: ""
---
<!-- This is an html comment and this won't appear in the rendered page. You are now editing the "content" area, the core of your description. Everything that you can do in markdown is allowed below. We added a couple of comments to guide your through documenting your progress. -->

## Project definition

### Background

Schizophrenia is a chronic psychiatric condition, frequently described as a dysconnectivity disorder, meaning symptoms that symptopms may reflect disrupted communication between brain regions, rather than dysfunction in one area. Positive symptoms, such as auditory verbal hallucinations (AVH), appear to involve a broader auditory-language brain network than the auditory cortex alone.

This network underpinning AVH includes the auditory cortex and superior temporal regions involved in speech perception, as well as frontal language regions and temporo-parietal regions involved in language production and integration. The Left Heschl's gyrus may be particularly important because studies have reported abnormal connectivity between this region and areas involved in language, memory, and self-monitoring.

For our project, we are using open-source neuroimaging data from OpenNeuro (ds004302) obtained from the article "Brain correlates of speech perception in schizophrenia patients with and without auditory hallucinations" by Soler-Vidal et al., 2022.

**Main Question**
Do AVH+, AVH-, and healthy control groups differ in how the auditory cortex communicates with other brain networks during speech perception?

**Research Objectives**
1. Learn resting-state fMRI preprocessing and quality control techniques 
2. Develop functional connectivity analysis and interpretation skills 
3. Apply preprocessing pipelines to schizophrenia neuroimaging data  
4. Conduct seed-based functional connectivity analyses   

<iframe width="560" height="315" src="https://www.youtube.com/embed/PTYs_JFKsHI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Tools

The dataset was obtained from OpenNeuro and organized according to the Brain Imaging Data Structure, or BIDS, which provides a standardized framework for managing neuroimaging data. Computationally intensive analyses were performed on SciNet, a high-performance computing cluster.

For preprocessing, we used fMRIPrep and FreeSurfer, while functional connectivity analyses were conducted in Python using Nilearn. Throughout the project, we used VS Code, Jupyter Notebooks, and the command line interface for coding and analysis, while GitHub was used for version control and project organization to support reproducibility and collaboration. 

**Procedure**
1. fMRI Preprocessing
2. Functional Connectivity
3. Fisher-z Transformation
4. Statistical Analysis Between Groups

### Data [to be updated]

[insert text]

### Deliverables [to be updated]

At the end of this project, we will have:
1. GitHub repository
2, README file
3. fMRI preprocessing script
4. Seed-based functional connectivity
5. Group-level FC script

## Results [to be updated]

### Progress overview 

Our project focuses on investigating the functional connectivity of the auditory cortex with regions implicated in auditory verbal hallucination in individuals with schizophrenia. Specifically, our region of interest was the left Heschl Gyrus.

### Tools we learned during this project [to be updated]

 * **Meta-project** P Bellec learned how to do a meta project for the first time, which is developing a framework while using it at the same time. It felt really weird, but somehow quite fun as well.
 * **Github workflow-** The successful use of this template approach will demonstrate that it is possible to incorporate dozens of students presentation on a website collaboratively over a few weeks.
 * **Project content** Through the project reports generated using the template, it is possible to learn about what exactly the brainhack school students are working on.

### Results [to be updated]

#### Deliverable 1: report template [to be updated]


#### Deliverable 2: project gallery [to be updated]

##### fMRI Analysis Pipeline [to be updated]

The repository of this project can be found [here](https://github.com/mtl-brainhack-school-2019/ecg_pupillometry_pipeline_kaufmann). The objective was to create a processing pipeline for ECG and pupillometry data. The motivation behind this task is that Marcel's lab (MIST Lab @ Polytechnique Montreal) was conducting a Human-Robot-Interaction user study. The repo features:
 * a [video introduction](http://www.youtube.com/watch/8ZVCNeX42_A) to the project.
 * a presentation [made in a jupyter notebook](https://github.com/mtl-brainhack-school-2019/ecg_pupillometry_pipeline_kaufmann/blob/master/BrainHackPresentation.ipynb) on the results of the project.
 * Notebooks for all analyses.
 * Detailed requirements files, making it easy for others to replicate the environment of the notebook.
 * An overview of the results in the markdown document.

#### Deliverable 3: [to be updated]



 ----
# OLD VERSION
# **Summary:** Investigate the functional connectivity of the auditory cortex with regions implicated in auditory verbal hallucination in individuals with schizophrenia.

# **Background** Schizophrenia is a chronic psychiatric condition, frequently described as a dysconnectivity disorder, meaning symptoms that symptopms may reflect disrupted communication between brain regions, rather than dysfunction in one area. Positive symptoms, such as auditory verbal hallucinations (AVH) appear to involve a broader auditory-language brain network than the auditory cortex alone. This network underpinning AVH includes auditory cortex and superior temporal regions involved in speech perception, as well as frontal language regions and temporo-parietal regions involved in language production and integration. The Left Heschl's gyrus may be particularly important because studies have reported abnormal connectivity between this regions and areas involved in language, memory, and self-monitoring. For our project, we are using open-source neuroimaging data from OpenNeuro obtained from the article "Brain correlates of speech perception in schizophrenia patients with and without auditory hallucinations" by Soler-Vidal et al., 2022.

# **Main Question** Do AVH+, AVH-, and healthy control groups differ in how the auditory cortex communicates with other brain networks during speech perception?

# **Research Objectives**
# 1. Learn resting-state fMRI preprocessing and quality control techniques 
# 2. Develop functional connectivity analysis and interpretation skills 
# 3. Apply preprocessing pipelines to schizophrenia neuroimaging data  
# 4. Conduct seed-based functional connectivity analyses    

# **Resources Used**
# The dataset was obtained from OpenNeuro and organized according to the Brain Imaging Data Structure, or BIDS, which provides a standardized framework for managing neuroimaging data. Computationally intensive analyses were performed on SciNet, a high-performance computing cluster.

# For preprocessing, we used fMRIPrep and FreeSurfer, while functional connectivity analyses were conducted in Python using Nilearn. Throughout the project, we used VS Code, Jupyter Notebooks, and the command line interface for coding and analysis, while GitHub was used for version control and project organization to support reproducibility and collaboration. 


# **Procedure**
# 1. fMRI Preprocessing
# 2. Functional Connectivity
# 3. Fisher-z Transformation
# 4. Statistical Analysis Between Groups

# **Deliverables**
# 1. GitHub repository
# 2, README file
# 3. fMRI preprocessing script
4. Seed-based functional connectivity
5. Group-level FC script



