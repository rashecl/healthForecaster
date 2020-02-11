# healthForecaster
Personalized predictions of health outcomes of lifestyle interventions 

# Set up the development environment: 

conda create -n insight python=3

conda activate insight

conda install altair black bqplot ipyvolume ipywebrtc ipywidgets jupyter jupyter_contrib_nbextensions mpld3 pandas pip pivottablejs qgrid scikit-learn scipy seaborn vega vega_datasets bokeh

conda install -c conda-forge nodejs

jupyter labextension install @jupyter-widgets/jupyterlab-manager

jupyter labextension install ipyvolume jupyter-threejs @jupyterlab/toc

jupyter labextension install jupyter-threejs bqplot qgrid

# Run the app: 

bokeh serve --show --log-level=debug HF.py
bokeh serve --show --log-level=debug HF_demo.py
