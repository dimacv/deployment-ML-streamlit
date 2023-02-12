# импортируем библиотеки pandas, numpy
import pandas as pd
import numpy as np
# импортируем библиотеку streamlit
import streamlit as st
# импортируем модуль pickle
import pickle

# загрузим модель стандартизации
with open('deployment_standardscaler.pkl', 'rb') as f:
    standardscaler = pickle.load(f)
# загрузим модель logreg
with open('deployment_logreg.pkl', 'rb') as f:
    logreg = pickle.load(f)

###  Функция запуска web интерфейса
def run():
    from PIL import Image
    image = Image.open('logo.jpeg')


    #st.image(image, use_column_width=False)
    st.sidebar.image(image)

    add_selectbox = st.sidebar.selectbox(
    "В каком режиме вы хотели сделать прогноз, Онлайн(Online) \nили загрузкой файла данных(Batch)?",
    ("Online", "Batch"))

    st.sidebar.info('Прогнозирования отклика на предложение автостраховки с использованием логистической регрессии')

    st.title("Прогнозирования отклика на предложение автостраховки:")

    if add_selectbox == 'Online':
        CusLV = st.number_input('Пожизненная ценность клиента [Customer Lifetime Value]')
        Income = st.number_input('Доход клиента [Income]', step=1)
        MonPAuto = st.number_input('Размер ежемесячной автостраховки [Monthly Premium Auto]', step=1)
        MonSLClaim = \
            st.number_input('Количество месяцев со дня подачи последнего страхового требования [Months Since Last Claim]', step=1)
        MonSPInception = \
            st.number_input('Количество месяцев с момента заключения страхового договора [Months Since Policy Inception]', step=1)
        NumComplaints = st.number_input('Количество открытых страховых обращений [Number of Open Complaints]', step=1)
        NumPolicies = st.number_input('Количество полисов [Number of Policies]', step=1)

        output = ""

        input_dict = {'Lifetime Value': CusLV,
                      'Income': Income,
                      'Monthly Premium Auto': MonPAuto,
                      'Months Since Last Claim': MonSLClaim,
                      'Months Since Policy Inception': MonSPInception,
                      'Number of Open Complaints': NumComplaints,
                      'Number of Policies': NumPolicies,
                      }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):

            input_df_standardscaled = standardscaler.transform(input_df)
            output = logreg.predict(input_df_standardscaled)
            output = str(output)

        st.success('Прогноз: {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Загрузите csv-файл с данными для осуществления прогнозов по ним:", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data_standardscaled = standardscaler.transform(data)
            predictions = logreg.predict(data_standardscaled)
            st.write(predictions)


if __name__ == '__main__':
    run()
