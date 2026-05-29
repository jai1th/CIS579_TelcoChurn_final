import streamlit as st
import pandas as pd
import webapp.predictors as predictors
import webapp.llm_server as llm_server
import time

def inputter():
    st.markdown("### Test Case Selection")
    test_option = st.selectbox("Choose a test scenario:", ("None", "Churn = False", "Churn = True"))
    
    # test case prefiller dropdown
    prefill = {}
    if test_option == "Churn = False":
        prefill = test_case_churn_false
    elif test_option == "Churn = True":
        prefill = test_case_churn_true

    predictor = []
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    # Senior Citizen, Partner, and Dependents
    for i, field in enumerate(bin_fields[:3]):
        with cols[i % 3]:
            default_val = prefill.get(field, False)
            predictor.append(st.selectbox(f"{field}:", ("Yes", "No"), index=0 if default_val else 1) == "Yes")

    # Tenure
    with col1:
        field = num_fields[0]
        predictor.append(st.number_input(f"{field} in months", value=float(prefill.get(field, 0.0)), step=1.0))

    # Remaining binary fields
    for i, field in enumerate(bin_fields[3:12]):
        with cols[i % 3]:
            default_val = prefill.get(field, False)
            predictor.append(st.selectbox(f"{field}:", ("Yes", "No"), index=0 if default_val else 1) == "Yes")

    # Remaining numeric fields
    for i, field in enumerate(num_fields[1:3]):
        with cols[i % 3]:
            predictor.append(st.number_input(f"{field} in USD", value=float(prefill.get(field, 0.0)), step=1.0))

    with col3:
        net_opt_map = {
            (True, False, False): "DSL",
            (False, True, False): "Fiber Optic",
            (False, False, True): "No"
        }
        current_key = tuple(prefill.get(k, False) for k in ["Internet Service_DSL", "Internet Service_Fiber optic", "Internet Service_No"])
        option = st.selectbox("Internet Service", ("DSL", "Fiber Optic", "No"), index=["DSL", "Fiber Optic", "No"].index(net_opt_map.get(current_key, "DSL")))
        predictor.extend([
            option == "DSL",
            option == "Fiber Optic",
            option == "No"
        ])

    with col2:
        con_opt_map = {
            (True, False, False): "Month-to-Month",
            (False, True, False): "One Year",
            (False, False, True): "Two Year"
        }
        current_key = tuple(prefill.get(k, False) for k in ["Contract_Month-to-month", "Contract_One year", "Contract_Two year"])
        option = st.selectbox("Contract Type", ("Month-to-Month", "One Year", "Two Year"), index=["Month-to-Month", "One Year", "Two Year"].index(con_opt_map.get(current_key, "Month-to-Month")))
        predictor.extend([
            option == "Month-to-Month",
            option == "One Year",
            option == "Two Year"
        ])

    with col3:
        pay_opt_map = {
            (True, False, False, False): "Bank Transfer (Automatic)",
            (False, True, False, False): "Credit Card (Automatic)",
            (False, False, True, False): "Electronic Check",
            (False, False, False, True): "Mailed Check"
        }
        current_key = tuple(prefill.get(k, False) for k in [
            "Payment Method_Bank transfer (automatic)",
            "Payment Method_Credit card (automatic)",
            "Payment Method_Electronic check",
            "Payment Method_Mailed check"
        ])
        option = st.selectbox("Payment Method", (
            "Bank Transfer (Automatic)",
            "Credit Card (Automatic)",
            "Electronic Check",
            "Mailed Check"
        ), index=[
            "Bank Transfer (Automatic)",
            "Credit Card (Automatic)",
            "Electronic Check",
            "Mailed Check"
        ].index(pay_opt_map.get(current_key, "Bank Transfer (Automatic)")))

        predictor.extend([
            option == "Bank Transfer (Automatic)",
            option == "Credit Card (Automatic)",
            option == "Electronic Check",
            option == "Mailed Check"
        ])

    return predictor

def submitter(predictor):
    with st.container():
        col1, col2 = st.columns([1,1000])

        with col2:
            if st.button("Predict Churn"):
                if len(predictor) == len(df.columns):
                    input_df = pd.DataFrame([predictor], columns=df.columns.to_list())
                    try:
                        prediction = predictors.predict_all(input_df)
                        prediction_conf = sum(prediction) * 100 / 3
                    except:
                        st.warning("###### Couldnt connect to ensemble. Try fixing the path in the predictors.py")
                    # The aim is to use ensembling to generate a confidence score of sorts to give the client about the prediction the system is making
                    if prediction_conf > 50:
                        print_pred = "The Customer will Churn.  \n Confidence: " + str(round(prediction_conf))   + "%"
                        st.warning(print_pred)
                        churn = True
                    else:
                        print_pred = "The Customer will Stay.  \n Confidence: " + str(100 - round(prediction_conf))  + "%"
                        st.success(print_pred)
                        churn = False


                    pred_txt = list(map(lambda x: "Churning" if x == 1 else "Not churning", prediction))
                    pred_df = pd.DataFrame([pred_txt], columns=['XG Classifier', 'Random Forest Classifier', 'Artificial Neural Network'])
                    st.table(pred_df)
                else:
                    st.error(f"Inputs incomplete ({len(predictor)}/{len(df.columns)} fields filled). Please fill out all fields.")
                try:
                    # Here we are turning to Gemma3:4b for advice on making services better or making steps to avoid churning
                    advice_df = input_df
                    advice_df['Churn'] = churn
                    container = st.container(border=True)
                    with st.spinner("Asking Gemma3:4b for advice...", show_time=True):
                        start = time.time()
                        advice = llm_server.advisor(advice_df)
                    container.markdown(advice)
                    st.markdown(f"*Completed the response in {round(time.time()-start)} seconds*")
                except:
                    st.warning("###### Couldnt connect to Ollama or the llm is not working. Try ***`ollama serve`*** before querying the advisor script")

if __name__ == '__main__':
    df_wo_churn = pd.read_csv('/media/jai/Projects/projects/ai-churn/CIS579_Churn/data/bs_eda_wo_index.csv')
    df = df_wo_churn.drop(columns=['Churn'])
    bin_fields = df.select_dtypes(include='bool').columns.to_list()
    num_fields = df.drop(columns=bin_fields).columns.to_list()

    # these are test cases to test the system and to save time during presentation
    test_case_churn_false = {
        'Senior Citizen': False,
        'Partner': True,
        'Dependents': True,
        'Tenure': 12,
        'Phone Service': True,
        'Multiple Lines': False,
        'Online Security': True,
        'Online Backup': False,
        'Device Protection': True,
        'Tech Support': True,
        'Streaming TV': False,
        'Streaming Movies': False,
        'Paperless Billing': False,
        'Monthly Charges': 30.0,
        'Total Charges': 360.0,
        'Internet Service_DSL': True,
        'Internet Service_Fiber optic': False,
        'Internet Service_No': False,
        'Contract_Month-to-month': False,
        'Contract_One year': True,
        'Contract_Two year': False,
        'Payment Method_Bank transfer (automatic)': False,
        'Payment Method_Credit card (automatic)': True,
        'Payment Method_Electronic check': False,
        'Payment Method_Mailed check': False
    }

    test_case_churn_true = {
        'Senior Citizen': True,
        'Partner': False,
        'Dependents': False,
        'Tenure': 1,
        'Phone Service': True,
        'Multiple Lines': True,
        'Online Security': False,
        'Online Backup': False,
        'Device Protection': False,
        'Tech Support': False,
        'Streaming TV': True,
        'Streaming Movies': True,
        'Paperless Billing': True,
        'Monthly Charges': 90.0,
        'Total Charges': 90.0,
        'Internet Service_DSL': False,
        'Internet Service_Fiber optic': True,
        'Internet Service_No': False,
        'Contract_Month-to-month': True,
        'Contract_One year': False,
        'Contract_Two year': False,
        'Payment Method_Bank transfer (automatic)': False,
        'Payment Method_Credit card (automatic)': False,
        'Payment Method_Electronic check': True,
        'Payment Method_Mailed check': False
    }
    st.set_page_config(layout="wide")
    col_left, col_right = st.columns([0.2,1])

    with col_left:
        st.image("webapp/assets/images/a51.png", width = 147)
    with col_right:
        st.title("Area 51's")
        st.markdown("## *Customer Churn Predictor*")

    col_left, col_spacer, col_right = st.columns([2, 0.5, 3])

    with col_left:
        st.header("Customer Details")
        user_input = inputter()
        st.markdown("### Ensemble model details:")
        ens_df = pd.read_csv("webapp/assets/tables/ens_metrics.csv")
        ens_met = pd.read_csv("webapp/assets/tables/metrics.csv")
        st.table(ens_df)
        st.table(ens_met)

    with col_right:
        st.header("Prediction Results")
        submitter(user_input)