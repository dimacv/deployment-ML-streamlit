# импортируем библиотеки pandas, numpy
import pandas as pd
import numpy as np
# импортируем библиотеку streamlit
import streamlit as st
# импортируем модуль dill
import dill

# функция предварительной подготовки
def preprocessing(df):

    # значения переменной age меньше 18 заменяем
    # минимально допустимым значением возраста
    df['age'] = np.where(df['age'] < 18, 18, df['age'])

    # создаем переменную Ratio - отношение количества
    # просрочек 90+ к общему количеству просрочек
    sum_of_delinq = (df['NumberOfTimes90DaysLate'] +
                     df['NumberOfTime30-59DaysPastDueNotWorse'] +
                     df['NumberOfTime60-89DaysPastDueNotWorse'])

    cond = (df['NumberOfTimes90DaysLate'] == 0) | (sum_of_delinq == 0)
    df['Ratio'] = np.where(
        cond, 0, df['NumberOfTimes90DaysLate'] / sum_of_delinq)

    # создаем индикатор нулевых значений переменной
    # NumberOfOpenCreditLinesAndLoans
    df['NumberOfOpenCreditLinesAndLoans_is_0'] = np.where(
        df['NumberOfOpenCreditLinesAndLoans'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # NumberRealEstateLoansOrLines
    df['NumberRealEstateLoansOrLines_is_0'] = np.where(
        df['NumberRealEstateLoansOrLines'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # RevolvingUtilizationOfUnsecuredLines
    df['RevolvingUtilizationOfUnsecuredLines_is_0'] = np.where(
        df['RevolvingUtilizationOfUnsecuredLines'] == 0, 'T', 'F')

    # преобразовываем переменные в категориальные, применив
    # биннинг и перевод в единый строковый формат
    for col in ['NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate']:
        df.loc[df[col] > 3, col] = 4
        df[col] = df[col].apply(lambda x: f"cat_{x}")

    # создаем список списков - список 2-факторных взаимодействий
    lst = [
        ['NumberOfDependents',
         'NumberOfTime30-59DaysPastDueNotWorse'],
        ['NumberOfTime60-89DaysPastDueNotWorse',
         'NumberOfTimes90DaysLate'],
        ['NumberOfTime30-59DaysPastDueNotWorse',
         'NumberOfTime60-89DaysPastDueNotWorse'],
        ['NumberRealEstateLoansOrLines_is_0',
         'NumberOfTimes90DaysLate'],
        ['NumberOfOpenCreditLinesAndLoans_is_0',
         'NumberOfTimes90DaysLate']
    ]

    # создаем взаимодействия
    for i in lst:
        f1 = i[0]
        f2 = i[1]
        df[f1 + ' + ' + f2 + '_interact'] = (df[f1].astype(str) + ' + '
                                             + df[f2].astype(str))

    # укрупняем редкие категории
    interact_columns = df.columns[df.columns.str.contains('interact')]
    for col in interact_columns:
        df.loc[df[col].value_counts()[df[col]].values < 55, col] = 'Other'

    return df


# загрузим сохраненную ранее модель
with open('pipeline_for_deployment.pkl', 'rb') as f:
    pipe = dill.load(f)

###  Функция запуска web интерфейса
def run():
    from PIL import Image
    image = Image.open('logo.jpeg')


    #st.image(image, use_column_width=False)
    st.sidebar.image(image)

    add_selectbox = st.sidebar.selectbox(
    "В каком режиме вы хотели сделать прогноз, Онлайн(Online) \nили загрузкой файла данных(Batch)?",
    ("Online", "Batch"))

    st.sidebar.info('Прогнозирования просрочки с использованием метода логистической регрессии.')

    st.title("Прогнозирования просрочки:")

    if add_selectbox == 'Online':
        RevolvingUtilizationOfUnsecuredLines = \
            st.number_input('Утилизация [RevolvingUtilizationOfUnsecuredLines]')
        age = st.number_input('Возраст клиента [age]', step=1)
        NumberOfTime30_59DaysPastDueNotWorse = \
            st.number_input('Количество просрочек 30-59 дней по данным БКИ [NumberOfTime30-59DaysPastDueNotWorse]',
                            step=1)
        DebtRatio = \
            st.number_input('Соотношение долга к доходу [DebtRatio]')
        MonthlyIncome = \
            st.number_input('Ежемесячный заработок [MonthlyIncome]')
        NumberOfOpenCreditLinesAndLoans = \
            st.number_input('Количество кредитов [NumberOfOpenCreditLinesAndLoans]', step=1)
        NumberOfTimes90DaysLate = \
            st.number_input('Количество просрочек 90+ по данным БКИ [NumberOfTimes90DaysLate]', step=1)
        NumberRealEstateLoansOrLines = \
            st.number_input('Количество ипотечных кредитов [NumberRealEstateLoansOrLines]', step=1)
        NumberOfTime60_89DaysPastDueNotWorse = \
            st.number_input('Количество просрочек 60-89 дней по данным БКИ [NumberOfTime60-89DaysPastDueNotWorse]', step=1)
        NumberOfDependents = st.number_input('Количество иждивенцев [NumberOfDependents]', step=1)
        output = ""

        input_dict = {'RevolvingUtilizationOfUnsecuredLines': RevolvingUtilizationOfUnsecuredLines,
                      'age': age,
                      'NumberOfTime30-59DaysPastDueNotWorse': NumberOfTime30_59DaysPastDueNotWorse,
                      'DebtRatio': DebtRatio,
                      'MonthlyIncome': MonthlyIncome,
                      'NumberOfOpenCreditLinesAndLoans': NumberOfOpenCreditLinesAndLoans,
                      'NumberOfTimes90DaysLate': NumberOfTimes90DaysLate,
                      'NumberRealEstateLoansOrLines': NumberRealEstateLoansOrLines,
                        'NumberOfTime60-89DaysPastDueNotWorse': NumberOfTime60_89DaysPastDueNotWorse,
                        'NumberOfDependents': NumberOfDependents
                      }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):

            # выполняем предварительную обработку новых данных
            input_df = preprocessing(input_df)

            # вычисляем вероятности для новых данных
            output = pipe.predict_proba(input_df)[:, 1]
            output = str(output)

        st.success('Вероятность просрочки: {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Загрузите csv-файл с новыми данными для вычисления вероятностей:", type=["csv"])

        if file_upload is not None:
            newdata = pd.read_csv(file_upload)
            # выполняем предварительную обработку новых данных
            newdata = preprocessing(newdata)

            # вычисляем вероятности для новых данных
            prob = pipe.predict_proba(newdata)[:, 1]

            # Вывод вероятностей на веб странице
            st.success('Вероятности просрочки для загруженных данных:')
            st.write(prob)


if __name__ == '__main__':
    run()
