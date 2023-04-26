
import streamlit as st
import pickle


with open("tfidf_vectorizer.sav", 'rb') as f:
    tfidf = pickle.load(f)

with open('svm_model.sav', 'rb') as f:
    svm_model = pickle.load(f)

st.title("Sentiment Analysis of Movie Reviews 📽️🍿")
new_review = st.text_input('Enter your review below', '')
print(new_review)


new_r = tfidf.transform([new_review])
# Make a prediction
prediction = svm_model.predict(new_r)
# Print the predicted sentiment label



if prediction == 0:
    if st.button('Show Sentiment'):
        st.write('The sentiment of this review is Negative')
        st.markdown("![Alt Text](https://media.giphy.com/media/0TMxedt3vbS8WXJMoH/giphy.gif)")
        
else:
    if st.button('Show Sentiment'):
        st.write('The sentiment of this review is Positive')
        st.markdown("![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzVkOTA3OGJlZTA5MDI5YWIxYzEyZDNlZWE2ZTFjOWNlMWZjZWVmNCZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/3ohfFsYe38TaL64fL2/giphy.gif)")
    



