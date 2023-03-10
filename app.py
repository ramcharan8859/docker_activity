import pickle
import streamlit as st


pickle_in = open("BankNote.pkl","rb")
Banknote=pickle.load(pickle_in)


def predict_note_authentication(variance,skewness,kurtosis,entropy):
    prediction=Banknote.predict([[variance,skewness,kurtosis,entropy]])
    print(prediction)
    return prediction

def main():
    st.title("class")
    variance = st.text_input("variance")
    skewness = st.text_input("skewness")
    kurtosis = st.text_input("kurtosis")
    entropy = st.text_input("entropy")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,kurtosis,entropy)
    st.success('The output is {}'.format(result))
   

if __name__=='__main__':
    main()