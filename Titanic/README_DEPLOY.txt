TITANIC STREAMLIT DEPLOYMENT FILES

1. Copy app.py and requirements.txt into the same folder as svc_trained_model.pkl.
2. In Jupyter, from that folder, test locally by running:
   %pip install streamlit
   !streamlit run app.py
   If the command does not open a browser, copy the local URL shown in the output.
3. For cloud deployment:
   - Create a GitHub repository.
   - Upload app.py, requirements.txt, and svc_trained_model.pkl to the repository.
   - Open https://share.streamlit.io/ and sign in with GitHub.
   - Create app; select the repository, branch, and app.py as the main file.
   - Deploy and copy the live URL to the top of your assignment notebook.

IMPORTANT:
The category mappings in app.py follow alphabetical LabelEncoder ordering and are consistent
with the sample encoded rows you shared. Before submission, compare these mappings with the
actual encoding/training cell in your notebook. The dataset is a 100-row teaching dataset;
describe the app as an educational demonstration, not a reliable real-world survival predictor.
