import pickle
import streamlit as st
import numpy as np

classifier = pickle.load(open('classifier.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

def prediction(a):
    Pension = 0.08 * a
    NHF = 0.025 * a
    Standard_relief = 0.2 * a
    Personal_relief = (200000/12) if a < (20000000/12) else 0.01* a
    Total_relief = Pension + NHF + Personal_relief + Standard_relief
    Net_taxable = a - Total_relief
    a = np.array([a,Pension,NHF,Standard_relief,Personal_relief,Total_relief,Net_taxable])
    a = scaler.transform(a.reshape(1,-1))

    prediction = classifier.predict(a)
    return prediction

def app():
    html_temp = """
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Net pay ML App</h1> 
    </div> 
    """

    st.markdown(html_temp,unsafe_allow_html = True)

    gross_pay = st.number_input('What is your gross pay')

    if st.button('Predict'):
        net_pay = prediction(gross_pay)
        st.success(f'Your net pay will be {net_pay}')

if __name__ == '__main__':
    app()