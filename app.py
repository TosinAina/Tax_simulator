import pickle
import streamlit as st
import numpy as np

grosspay_regressor = pickle.load(open('grosspay_regressor.pkl','rb'))
grosspay_scaler = pickle.load(open('grosspay_scaler.pkl','rb'))
netpay_regressor = pickle.load(open('netpay_regressor.pkl','rb'))
netpay_scaler = pickle.load(open('netpay_scaler.pkl','rb'))

def netpay_prediction(a):
    Pension = 0.08 * a
    NHF = 0.025 * a
    Standard_relief = 0.2 * a
    Personal_relief = (200000/12) if a < (20000000/12) else 0.01* a
    Total_relief = Pension + NHF + Personal_relief + Standard_relief
    Net_taxable = a - Total_relief
    a = np.array([a,Pension,NHF,Standard_relief,Personal_relief,Total_relief,Net_taxable])
    a = netpay_scaler.transform(a.reshape(1,-1))
    netpay = netpay_regressor.predict(a)
    return netpay


def grosspay_prediction(a):
    a = np.array(a)
    a = grosspay_scaler.transform(a.reshape(1,-1))
    grosspay = grosspay_regressor.predict(a)
    return grosspay

def app():

    menu = ['Predict Netpay','Predict Grosspay']
    options = st.sidebar.selectbox("Menu",menu)

    if options == 'Predict Netpay':
        html_temp = """
        <div style ="background-color:blue;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Streamlit Net pay ML App</h1> 
        </div> 
        """

        st.markdown(html_temp,unsafe_allow_html = True)

        gross_pay = st.number_input('What is your gross pay')

        if st.button('Predict'):
            net_pay = netpay_prediction(gross_pay)
            st.success(f'Your net pay will be {net_pay}')

    else:
        html_temp = """
        <div style ="background-color:blue;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Streamlit Gross pay ML App</h1> 
        </div> 
        """

        st.markdown(html_temp,unsafe_allow_html = True)

        net_pay = st.number_input('What is your net pay')

        if st.button('Predict'):
            gross_pay = grosspay_prediction(net_pay)
            st.success(f'Your gross pay will be {gross_pay}')

if __name__ == '__main__':
    app()