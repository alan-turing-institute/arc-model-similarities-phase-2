# Analysis
The scripts and notebooks in this folder are for generating images and plots used in the final report.

The data for images has not been stored here, due to the size of the files. One exception is the wandb output file, which is saved in the data folder.

To replicate the images and plots, the file in `scripts\create_data.py` will need to be run. The user may want to change the `device` variable, which determines whether the inception embeddings will be caluclated on cpu or gpu.

The images and plots used in the report can then be replicated by running the following notebooks:
- `hypothesis1_2.ipynb` - Plots to analyse Hypotheses 1 and 2
- `hypothesis3.ipynb` - Plots to analyse Hypothesis 3
- `hypothesis4.ipynb` - Plots to analyse Hypothesis 4
- `viewing_images.ipynb` - Notebook that produces a view of three images from CIFAR-10 and the corresponding transforms used in the work
- `viewing_embeddings.ipynb` - Notebook that produces heatmaps of the UMAP embeddings (2D) by number of records dropped and by transform
- `viewing_pixel_values.ipynb` - Notebook for analysing the distribution of six pixel values across the whole training dataset in CIFAR-10

All images and plots are stored in the `output` folder
