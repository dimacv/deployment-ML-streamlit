# deployment-ML-streamlit

## Запуск локально: 

##### установка зависимостей
pip3 install -r requirements.txt   

##### Собственно запуск  
streamlit run app.py --server.port=8501 --server.address=0.0.0.0    
   
------------------------------------------------------------------  

### Сборка докера:
docker build -t streamlit .  
  
### Запуск контейнера: 
docker run -p 8501:8501 --name streamlit streamlit 

