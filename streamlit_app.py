# streamlit_app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Climate-Smart Beef Production Dashboard", layout="wide")

import streamlit as st, base64, pathlib

def set_background(local_img_path: str):
    img_bytes = pathlib.Path(local_img_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    st.markdown(f"""
    <style>
    /* Main page background */
    [data-testid="stAppViewContainer"] > .main {{
        background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
    }}
    /* Sidebar background (optional) */
    [data-testid="stSidebar"] > div:first-child {{
        background: url("data:image/png;base64,{b64}") no-repeat center center;
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

# call this near the top of your script
# set_background("dash_background_3.jpeg")   # <-- put your image path here

# ------------------------------
# Helper functions (gauges etc.)
# ------------------------------
def _color(value, pre):
    if value < pre:
        return 'green'
    elif value == pre:
        return 'orange'
    else:
        return 'red'

def current_gauge_chart(value, title):
    # dynamic range around the current value, similar to original
    min_val = -(1 + (2 * value // 50)) * 50
    max_val =  (1 + (2 * value // 50)) * 50
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={'suffix': ' MMT'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f'<b>{title}</b>', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': 'orange'},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': float(value)
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def post_gauge_chart(pre, value, title):
    # range anchored to the baseline ("pre")
    min_val = -(1 + (2 * pre // 50)) * 50
    max_val =  (1 + (2 * pre // 50)) * 50
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={'suffix': ' MMT'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f'<b>{title}</b>', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': _color(value, pre)},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': float(value)
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def money_gauge_chart(value, title):
    min_val, max_val = -1000, 1000
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={'suffix': '$'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f'<b>{title}</b>', 'font': {'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': 'green' if value >= 0 else 'red'},
        }
    ))
    fig.update_layout(height=300)
    return fig

# ------------------------------
# Model + preprocessing
# ------------------------------
interventions = [
    'Cover crops (Corn)', 'Cover crops (SB)', 'Irrigation Efficiency (Corn)', 'Climate-adapted plants',
    'DMI', 'GE', 'grass/grain', 'ADF', 'NDF', 'CP', 'Tannins', '3-NOP', 'Red sea weed',
    'Nitrate', 'Silvo-pasture', 'Riparian restoration', 'AMP', 'Manure management',
    'Solar', 'Energy efficiency', 'Genetics'
]
levels = ['High', 'Low', 'Medium']  # keep original order

def preprocess_input(input_data: dict) -> pd.DataFrame:
    # one-hot columns in the same order as the Dash app
    cols = [f"{intervention}_{level}" for intervention in interventions for level in levels]
    one_hot_df = pd.DataFrame(0, index=[0], columns=cols)
    for intervention, level in input_data.items():
        col = f"{intervention}_{level}"
        if col in one_hot_df.columns:
            one_hot_df.at[0, col] = 1
    return one_hot_df

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("linear_regression_model.pkl")
    except Exception as e:
        st.warning(f"Model file 'linear_regression_model.pkl' not found or failed to load: {e}. "
                   "Predictions will be disabled (showing baseline only).")
        return None

linear_regression_model = load_model()

# -----------------------------------
# Session state (defaults + utilities)
# -----------------------------------
def init_state():
    defaults = {
        "Number of cattle": 100,
        "Length of Production": 12,
        "Age": 12,
        "Breed": "Brahman",
        "Sex": "Female",
        "Health": "Healthy",
        "CarbonCredit": 10.0,
        "production": None,
        "base_emissions": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()



# -----------------------------------
# Common input widgets (Home summary)
# -----------------------------------
def home_inputs():
    st.subheader("Animal & Scenario Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["Number of cattle"] = st.number_input(
            "Number of cattle", min_value=0, value=int(st.session_state["Number of cattle"]), step=1
        )
        st.session_state["Age"] = st.number_input(
            "Age (months)", min_value=0, value=int(st.session_state["Age"]), step=1
        )
        st.session_state["CarbonCredit"] = st.number_input("Carbon credit rate ($/MMT)",
                                                           value=float(st.session_state["CarbonCredit"]), step=1.0)
    with col2:
        st.session_state["Length of Production"] = st.number_input(
            "Length of Production (months)", min_value=0, value=int(st.session_state["Length of Production"]), step=1
        )
        st.session_state["Breed"] = st.selectbox(
            "Breed", ["Brahman", "Hereford", "Charloais", "Angus", "Crossbred", "Other"],
            index=["Brahman","Hereford","Charloais","Angus","Crossbred","Other"].index(st.session_state["Breed"])
        )
    with col3:
        st.session_state["Sex"] = st.selectbox("Sex", ["Male", "Female"],
                                               index=["Male","Female"].index(st.session_state["Sex"]))
        st.session_state["Health"] = st.selectbox("Health status", ["Sick", "Healthy", "Unknown"],
                                                  index=["Sick","Healthy","Unknown"].index(st.session_state["Health"]))

    st.caption("These settings feed into baseline emissions and credits calculations.")

# -----------------------------------
# Emissions baseline calculation
# -----------------------------------
def compute_baseline_emissions():
    noc   = st.session_state["Number of cattle"]
    lop   = st.session_state["Length of Production"]
    age   = st.session_state["Age"]
    breed = st.session_state["Breed"]
    sex   = st.session_state["Sex"]
    health= st.session_state["Health"]

    breed_change = {'Brahman': 0, 'Angus': 0.05, 'Hereford': -0.05, 'Charloais': 0.03, 'Crossbred': -0.01, 'Other': 0}
    sex_change   = {'Male': 0.03, 'Female': -0.03}
    health_change= {'Sick': 0.5, 'Healthy': 0, 'Unknown': 0}

    base_emissions = 2.0
    change_rate = 1.0

    if age <= 8:
        change_rate += 0.1
    if age >= 18:
        change_rate -= 0.1

    change_rate += breed_change.get(breed, 0) + sex_change.get(sex, 0) + health_change.get(health, 0)

    emissions = base_emissions * change_rate * (lop / 12.0) * noc
    st.session_state["base_emissions"] = emissions
    return emissions

# -----------------------------------
# Cow-Calf layout (sliders + gauges)
# -----------------------------------
def cow_calf_layout():

    # Baseline emissions from current animal inputs
    emissions = compute_baseline_emissions()

    # Top row: params + BAU gauge + (we'll add New Scenario & Credits later)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown("### Animal Parameters (from Home)")
        st.write(f"**Number of cattle:** {st.session_state['Number of cattle']}")
        st.write(f"**Length of Production:** {st.session_state['Length of Production']} months")
        st.write(f"**Age:** {st.session_state['Age']} months")
        st.write(f"**Breed:** {st.session_state['Breed']}")
        st.write(f"**Sex:** {st.session_state['Sex']}")
        st.write(f"**Health:** {st.session_state['Health']}")

    with colB:
        st.plotly_chart(current_gauge_chart(emissions, "Business as usual"), use_container_width=True)

    # ---------------------------
    # Interventions & Practices
    # ---------------------------
    st.markdown("---")
    st.markdown("<h3 style='text-align:center;'>Interventions & Practices</h3>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        st.markdown("#### Feed Production")
        ccc = st.slider("Cover crops (Corn)", 0, 2, 0, 1, key="ccc")
        iec = st.slider("Irrigation efficiency (Corn)", 0, 2, 0, 1, key="iec")
        ccs = st.slider("Cover crops (SB)", 0, 2, 0, 1, key="ccs")
        cap = st.slider("Climate-adapted plants", 0, 2, 0, 1, key="cap")

        st.markdown("#### Feeds & Feeding")
        dmi = st.slider("DMI", 0, 2, 0, 1, key="dmi")
        adf = st.slider("ADF", 0, 2, 0, 1, key="adf")
        ge  = st.slider("GE",  0, 2, 0, 1, key="ge")

    with c2:
        st.markdown("#### Feeds & Feeding")
        ndf = st.slider("NDF", 0, 2, 0, 1, key="ndf")
        ddg = st.slider("grass/grain", 0, 2, 0, 1, key="ddg")
        cp  = st.slider("CP",  0, 2, 0, 1, key="cp")

        st.markdown("#### Feed additives")
        tannins = st.slider("Tannins", 0, 2, 0, 1, key="tannins")
        nop     = st.slider("3-NOP",   0, 2, 0, 1, key="nop")

    with c3:
        st.markdown("#### Feed additives")
        rsw     = st.slider("Red sea weed", 0, 2, 0, 1, key="rsw")
        nitrate = st.slider("Nitrate",      0, 2, 0, 1, key="nitrate")

        st.markdown("#### Farm management")
        silvo = st.slider("Silvo-pasture",        0, 2, 0, 1, key="silvo")
        rr    = st.slider("Riparian restoration", 0, 2, 0, 1, key="rr")

    with c4:
        st.markdown("#### Farm management")
        amp    = st.slider("AMP",               0, 2, 0, 1, key="amp")
        manure = st.slider("Manure management", 0, 2, 0, 1, key="manure")

        st.markdown("#### Renewable energy")
        solar = st.slider("Solar",             0, 2, 0, 1, key="solar")
        ee    = st.slider("Energy efficiency", 0, 2, 0, 1, key="ee")
        gen   = st.slider("Genetics",          0, 2, 0, 1, key="genetics")

    to_level = ["Low", "Medium", "High"]

    input_data = {
        "Cover crops (Corn)":           to_level[ccc],
        "Irrigation Efficiency (Corn)": to_level[iec],
        "Cover crops (SB)":             to_level[ccs],
        "Climate-adapted plants":       to_level[cap],
        "DMI":                          to_level[dmi],
        "ADF":                          to_level[adf],
        "GE":                           to_level[ge],
        "NDF":                          to_level[ndf],
        "grass/grain":                  to_level[ddg],
        "CP":                           to_level[cp],
        "Tannins":                      to_level[tannins],
        "3-NOP":                        to_level[nop],
        "Red sea weed":                 to_level[rsw],
        "Nitrate":                      to_level[nitrate],
        "Silvo-pasture":               to_level[silvo],
        "Riparian restoration":         to_level[rr],
        "AMP":                          to_level[amp],
        "Manure management":           to_level[manure],
        "Solar":                        to_level[solar],
        "Energy efficiency":            to_level[ee],
        "Genetics":                     to_level[gen],
    }

    any_changed = any(v != 0 for v in [
        ccc, iec, ccs, cap, dmi, adf, ge, ndf, ddg, cp,
        tannins, nop, rsw, nitrate, silvo, rr, amp, manure, solar, ee, gen
    ])



    # Predict / compute new scenario
    emissions_post = emissions
    prediction_pct = 0.0
    if any_changed and linear_regression_model is not None:
        try:
            X = preprocess_input(input_data)
            prediction_pct = float(linear_regression_model.predict(X)[0])  # % change
            emissions_post = emissions * (1.0 + prediction_pct / 100.0)
        except Exception as e:
            st.warning(f"Prediction failed: {e}. Using baseline for 'New scenario'.")

    # Carbon credits
    cc_rate = float(st.session_state.get("CarbonCredit", 0.0))
    carbon_credits = (
        -emissions * (prediction_pct / 100.0) * cc_rate
        if any_changed and linear_regression_model is not None
        else 0.0
    )

    # Gauges for New Scenario & Carbon Credits
    with colC:
        st.plotly_chart(post_gauge_chart(emissions, emissions_post, "New scenario"), use_container_width=True)
    with colD:
        st.plotly_chart(money_gauge_chart(carbon_credits, "Carbon credits"), use_container_width=True)

# ------------------------------
# Home page
# ------------------------------
def home_page():
    st.markdown(
        "<h1 style='text-align:center;color:#007bff;'>Climate-Smart Beef Production Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='text-align:center;'>Empowering Decisions for a Sustainable Future!</h2>",
        unsafe_allow_html=True
    )

    home_inputs()

    st.markdown("### Select Production Type")
    col1, col2, col3 = st.columns(3)
    if col1.button("Cow-Calf"):
        st.session_state["production"] = "Cow-Calf"
    if col2.button("Stocker"):
        st.session_state["production"] = "Stocker"
    if col3.button("Feedlot"):
        st.session_state["production"] = "Feedlot"

    if st.session_state["production"]:
        st.success(f"Selected: {st.session_state['production']}")
        cow_calf_layout()

# ------------------------------
# Router
# ------------------------------
home_page()