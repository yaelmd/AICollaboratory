# AICollaboratory :globe_with_meridians:
Instance level data extracted from BigBench to integrate in [AICollaboratory](https://ai-collaboratory.jrc.ec.europa.eu/)


## What is AICollaboratory? 💡
[AICollaboratory](https://ai-collaboratory.jrc.ec.europa.eu/) is a tool to analyse, evaluate, compare and monitor the state-of-the art of Artificial Intelligence systems. This project provides an unifying setting that incorporates data, knowledge and measurements to characterise AI systems. The AIcollaboratory is framed in the context of AI watch, a knowledge service of the European Commission carried out by the JRC in collaboration with DG CNECT. 

## Data :page_facing_up:

[Data](https://upvedues-my.sharepoint.com/:f:/g/personal/ymordav_upv_edu_es/Ek3OQMpn9c1LpIqdhXO9STkBVEs2czgqo5MuMelIV1LLUA?e=niwklJ) (OneDrive link) contains a folder for each multiple choice task in [BigBench](https://github.com/google/BIG-bench). Inside each folder there are 4 files with the obtained data for 0,1,2 and 3 shot. They follow the next structure:

- Input
- Targets
- Scores
- Target values
- Correct
- Absolute scores
- Normalized scores
- Metrics
- Model name: [BIG-G sparse, BIG-G T=0]
- Model family: [1b, 2b, 2m, 4b, 8b, 16m, 27b, 53m, 125m, 128b, 244m, 422m] 
- Task
- Shot

## How to use the code 💻
Running the file [getdata](../main/code/getdata.py), a csv is obtained with the selected data. Changing the parameters of LogLoader makes possible to choose different tasks, model families, number of shots, etc.

⚠️Warning: this code handles log files from [BigBench](https://github.com/google/BIG-bench) tasks, which are not included in the repository because of their size.
