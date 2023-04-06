# Deploying a machine learning model (complex scikit-learn pipeline) using Streamlit.

## Run locally:

##### installing dependencies
``` pip3 install -r requirements.txt ```   

##### Actually launch  
``` streamlit run app.py --server.port=8501 --server.address=0.0.0.0 ```    
   
------------------------------------------------------------------  

## Run in docker:

### Docker build:

``` docker build -t streamlit . ```  
  
### Starting a named container in daemon mode on port 8501 :

``` docker run -rm -p 8501:8501 --name streamlit -d streamlit ``` 

