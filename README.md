# twin-stars
Code repository for study of twin star detection prospects with population-scale gravitational wave observations

Notebook:

SimulateTwinStarInference.ipynb

Script:

SimulateTwinStarInference.py scenario base_eos hybrid_eos scenario_tag version_tag

Batch:

bash SimulateTwinStarInference_DAG.sh $(cat ./SimulateTwinStarInference_DAG.in)
condor_submit_dag ./batch/SimulateTwinStarInference.dag