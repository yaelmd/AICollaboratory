# AICollaboratory :globe_with_meridians:
Instance level data extracted from BigBench to integrate in [AICollaboratory](https://ai-collaboratory.jrc.ec.europa.eu/)


## What is AICollaboratory? 💡
[AICollaboratory](https://ai-collaboratory.jrc.ec.europa.eu/) is a tool to analyse, evaluate, compare and monitor the state-of-the art of Artificial Intelligence systems. This project provides an unifying setting that incorporates data, knowledge and measurements to characterise AI systems. The AIcollaboratory is framed in the context of AI watch, a knowledge service of the European Commission carried out by the JRC in collaboration with DG CNECT. 

## Data



## How to use the code 💻
Running the file [getdata](../main/code/getdata.py), a csv is obtained with the selected data. Changing the parameters of LogLoader makes possible to choose different tasks, model families, number of shots, etc.

⚠️Warning: this code handles log files from [BigBench](https://github.com/google/BIG-bench) tasks, which are not included in the repository because of their size.
