import streamlit as st


# Streamlit app
st.title("Toxicity Classification App")

# Text input for user comment
user_input = st.text_area("Enter your comment:")

if user_input:
    # Clean the user input
    cleaned_input = clean_text(user_input)

    # Vectorize the cleaned input (replace with your actual vectorizer)
    vectorizer = TfidfVectorizer()  # Initialize your vectorizer
    input_vect = vectorizer.transform([cleaned_input])

    # Make predictions for each label
    predictions = {}
    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        prediction = models[label].predict_proba(input_vect)[:, 1][0]
        predictions[label] = prediction

    # Display predictions
    st.subheader("Toxicity Predictions:")
    for label, prediction in predictions.items():
        st.write(f"{label}: {prediction:.2f}")
