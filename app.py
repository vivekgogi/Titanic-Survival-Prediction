import streamlit as st
import pandas as pd
import pickle

# Set Page Layout as Wide
# st. set_page_config(layout="wide")

# Load the pickled model
model = pickle.load(open('rf_clf_titanic.pkl', 'rb'))

# Create the web application using Streamlit


def main():
    st.title("Titanic Survival Prediction")
    st.markdown(
        "Enter the pasenger details to predict survival on the Titanic.")

    # Create input fields for passenger details
    st.header("Passenger Details")

    title = st.selectbox(
        "Name Title", ['Mr', 'Miss', 'Master', 'Rare'])

    column1, column2 = st.columns(2)

    with column1:
        age = st.number_input("Age", min_value=0, max_value=100, value=0)
    with column2:
        sex = st.selectbox("Sex", ['Male', 'Female'])

    column3, column4 = st.columns(2)

    with column3:
        siblings_spouses = st.number_input(
            "Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    with column4:
        parents_children = st.number_input(
            "Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)

    total_fare = st.number_input(
        "Total Fare", min_value=0, max_value=500, value=0)

    column5, column6, column7 = st.columns(3)

    with column5:
        embarked = st.selectbox(
            "Embarked", ['Cherbourg', 'Queenstown', 'Southampton'])
    with column6:
        passenger_class = st.selectbox("Passenger Class", [1, 2, 3])
    with column7:
        deck = st.selectbox(
            "Deck", ['Not Sure', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])

    family_members = siblings_spouses + parents_children + 1

    family_size = None
    if family_members == 1:
        family_size = 'singleton'
    elif family_members > 1 and family_members < 5:
        family_size = 'small'
    else:
        family_size = 'large'

    individual_fare = total_fare / family_members

    # Mapping of categorical values to numerical values
    title_mapping = {
        'Mr': 1,
        'Miss': 2,
        'Master': 3,
        'Rare': 4
    }
    title = title_mapping[title]

    sex_mapping = {
        'Male': 0,
        'Female': 1
    }
    sex = sex_mapping[sex]

    embarked_mapping = {
        'Cherbourg': 0,
        'Queenstown': 1,
        'Southampton': 2
    }
    embarked = embarked_mapping[embarked]

    deck_mapping = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'T': 7,
        'Not Sure': 8,
    }
    deck = deck_mapping[deck]

    family_size_mapping = {
        'large': 0,
        'singleton': 1,
        'small': 2
    }
    family_size = family_size_mapping[family_size]

    # Prepare input data
    passenger_data = {
        'Pclass': passenger_class,
        'Sex': sex,
        'Age': age,
        'SibSp': siblings_spouses,
        'Parch': parents_children,
        'Fare': total_fare,
        'Embarked': embarked,
        'Deck': deck,
        'Title': title,
        'family_members': family_members,
        'family_size': family_size,
        'individual_fare': individual_fare
    }

    # Create a dataframe from the input data
    df = pd.DataFrame(passenger_data, index=[0])

    if st.button("Predict Survival"):
        if title == -1 or embarked == -1 or deck == -1:
            st.error("Please fill in all the required fields.")
        if not df.empty and not df.isnull().values.any():
            prediction = model.predict(df)
            if prediction[0] == 1:
                st.success(f"The passenger is predicted to have SURVIVED.")
            else:
                st.error(f"The passenger is predicted to have NOT SURVIVED.")
        else:
            st.error("Please fill in all the required fields.")


# Run the web application
if __name__ == '__main__':
    main()
