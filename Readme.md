## Project setup

1. open your terminal and enter command

   > conda create -p venv python=3.10
   > conda activate venv/

2. make a .env file and store your gemini api key
   GOOGLE_API_KEY = "YOUR_API_KEY"

3. installing the requirements of your project

   > pip install -r requirements.txt

4. run your app
   > streamlit run app.py

## Troubleshooting

1. check if you have done the installations in the activated conda environment
2. if you get errors related to streamlit package
   > pip install --upgrade streamlit
3. if you get errors with other packages, you can try out installing upgraded packages using the syntax as above
