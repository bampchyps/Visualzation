import streamlit as st
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
import folium
import plotly.express as px
from streamlit_folium import st_folium
# Load the model
model = joblib.load("model.joblib")

# List of stopwords
stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

# Streamlit UI
st.title('Address Detection Visualization')
#st.markdown('------------------------------------')
# Description
# Description
st.markdown("""
    <div style='border: 1px solid black; padding: 10px; border-radius: 5px; background-color: #F9F9F9;'>
        Visualization นี้แสดงถึงประสิทธิภาพของแบบจำลองที่ใช้ระบุและจัดประเภทคำหรือวลีในข้อความที่เกี่ยวข้องกับชื่อ - นามสกุลและที่อยู่ในหลากหลายรูปแบบ โดยให้เลือกรูปแบบการใส่ที่อยู่ในแบบต่างๆ เพื่อดูประสิทธิภาพของแบบจำลองดังนี้ เลือกจังหวัดที่อยากทราบประสิทธิภาพของแบบจำลอง
    </div>
""", unsafe_allow_html=True)

st.markdown('------------------------------------')
# Feature extraction function
def tokens_to_features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word.word": word,
        "word[:3]": word[:3],
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,
        "word.isdigit()": word.isdigit(),
        "word.islen5": word.isdigit() and len(word) == 5
    }
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
            "-1.word.isspace()": prevword.isspace(),
            "-1.word.is_stopword()": prevword in stopwords,
            "-1.word.isdigit()": prevword.isdigit(),
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
            "+1.word.isspace()": nextword.isspace(),
            "+1.word.is_stopword()": nextword in stopwords,
            "+1.word.isdigit()": nextword.isdigit(),
        })
    else:
        features["EOS"] = True
    return features

# Parsing function
def parse(text):
    tokens = text.split()
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
    return model.predict([features])[0]

# Province list for multi-select
provinces = [
    "กรุงเทพ", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "ชลบุรี", "เชียงใหม่", "ภูเก็ต"
]


# List of provinces in Thailand with coordinates
province_coords = {
    "กรุงเทพ": (13.7563, 100.5018),
    "นนทบุรี": (13.8621, 100.5144),
    "ปทุมธานี": (14.0200, 100.5250),
    "สมุทรปราการ": (13.5991, 100.5990),
    "ชลบุรี": (13.3611, 100.9847),
    "เชียงใหม่": (18.7883, 98.9853),
    "ภูเก็ต": (7.8804, 98.3923),
    "เชียงราย": (19.9100, 99.8400),
    "ลำพูน": (18.5800, 99.0000),
    "ลำปาง": (18.2888, 99.4900),
    "แพร่": (18.1317, 100.2024),
    "น่าน": (18.7833, 100.7836),
    "อุตรดิตถ์": (17.6238, 100.0993),
    "ตาก": (16.8791, 99.1256),
    "สุโขทัย": (17.0148, 99.8260),
    "พิษณุโลก": (16.8219, 100.2659),
    "พิจิตร": (16.4422, 100.3480),
    "กำแพงเพชร": (16.4720, 99.5210),
    "นครสวรรค์": (15.7007, 100.1372),
    "อุทัยธานี": (15.3829, 100.0269),
    "ชัยนาท": (15.1877, 100.1253),
    "ลพบุรี": (14.7995, 100.6534),
    "สิงห์บุรี": (14.8911, 100.3984),
    "อ่างทอง": (14.5896, 100.4526),
    "พระนครศรีอยุธยา": (14.3704, 100.5853),
    "สระบุรี": (14.5289, 100.9109),
    "นครนายก": (14.2061, 101.2135),
    "ฉะเชิงเทรา": (13.6904, 101.0762),
    "ปราจีนบุรี": (14.0516, 101.3682),
    "สระแก้ว": (13.8240, 102.0644),
    "ระยอง": (12.6827, 101.2570),
    "จันทบุรี": (12.6077, 102.1110),
    "ตราด": (12.2438, 102.5156),
    "กาญจนบุรี": (14.0041, 99.5483),
    "ราชบุรี": (13.5360, 99.8177),
    "เพชรบุรี": (13.1111, 99.9391),
    "ประจวบคีรีขันธ์": (11.8115, 99.7970),
    "นครปฐม": (13.8198, 100.0638),
    "สมุทรสาคร": (13.5472, 100.2744),
    "สมุทรสงคราม": (13.4115, 100.0005),
    "สุพรรณบุรี": (14.4753, 100.1160),
    "กาฬสินธุ์": (16.4338, 103.5062),
    "ขอนแก่น": (16.4322, 102.8236),
    "ชัยภูมิ": (15.8057, 102.0310),
    "นครพนม": (17.4098, 104.7784),
    "นครราชสีมา": (15.0000, 102.1167),
    "บุรีรัมย์": (14.9930, 103.1029),
    "มหาสารคาม": (16.1808, 103.3000),
    "มุกดาหาร": (16.5453, 104.7233),
    "ยโสธร": (15.7944, 104.1452),
    "ร้อยเอ็ด": (16.0527, 103.6530),
    "ศรีสะเกษ": (15.1184, 104.3299),
    "สกลนคร": (17.1552, 104.1388),
    "สุรินทร์": (14.8832, 103.4935),
    "หนองคาย": (17.8782, 102.7421),
    "หนองบัวลำภู": (17.2046, 102.4404),
    "อุดรธานี": (17.4138, 102.7874),
    "อุบลราชธานี": (15.2444, 104.8475),
    "อำนาจเจริญ": (15.8683, 104.6283),
    "ยะลา": (6.5400, 101.2800),
    "ปัตตานี": (6.8689, 101.2501),
    "นราธิวาส": (6.4254, 101.8253),
    "สตูล": (6.6237, 100.0673),
    "สงขลา": (7.1894, 100.5951),
    "พัทลุง": (7.6173, 100.0802),
    "ตรัง": (7.5580, 99.6117),
    "นครศรีธรรมราช": (8.4304, 99.9631),
    "กระบี่": (8.0863, 98.9063),
    "พังงา": (8.4510, 98.5310),
    "สุราษฎร์ธานี": (9.1416, 99.3296),
    "ชุมพร": (10.4959, 99.1800)
}

provinces = list(province_coords.keys())

# Streamlit multi-select for provinces
selected_provinces = st.multiselect(
    "เลือกจังหวัด",
    provinces,
    default=["กรุงเทพ", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "ชลบุรี", "เชียงใหม่", "ภูเก็ต"]
)

# Function to create a map and highlight selected provinces
def create_map(selected_provinces):
    # Initialize map centered in Thailand
    m = folium.Map(location=[13.736717, 100.523186], zoom_start=6)

    # Add markers for selected provinces
    for province in selected_provinces:
        if province in province_coords:
            folium.Marker(
                location=province_coords[province],
                popup=f"{province}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
    
    return m

# Display the map
st.markdown("### ตำแหน่งจังหวัดที่เลือก")
province_map = create_map(selected_provinces)
st_folium(province_map, width=800, height=600)

# Address format options with "No prefix" set as the default
format_options = {
    "ชื่อ": ["นาย", "นาง", "นางสาว","ด.ช.","ด.ญ.", "ไม่มี"],
    "หมู่บ้าน": ["หมู่บ้าน", "ม.", "ไม่มี"],
    "ซอย": ["ซอย", "ซ.", "ไม่มี"],
    "ถนน": ["ถนน", "ถ.", "ไม่มี"],
    "ตำบล": ["ตำบล", "ต.", "แขวง", "ไม่มี"],
    "อำเภอ": ["อำเภอ", "อ.", "เขต", "ไม่มี"],
    "จังหวัด": ["จังหวัด", "จ.", "ไม่มี"]
}

# Set "No prefix" as the default for all components
selected_formats = {
    key: st.multiselect(
        f"เลือกรูปแบบคำนำหน้า {key}",
        options,
        default=["ไม่มี"] if "ไม่มี" in options else []
    )
    for key, options in format_options.items()
}

# Mock data for other components
first_names = ["สมชาย", "สมหญิง", "วรพล", "จันทร์เพ็ญ","เข็มกลัด", "กรการ", "เทิดทูล", "เด่นชัด"]
village_variants = ["สุขสันต์", "ทรัพย์มั่นคง","บ้านเพชร", "บ้านทองคำ", "บ้านร่มรื่น", "บ้านฟ้าใส"]
soi_variants = ["โชคชัย", "เจริญนคร", "สุขสมใจ", "เจริญรุ่งเรือง", "บ้านใหม่", "สุขใจ", "เจริญนคร"]
road_variants = ["บางนา-ตราด", "เพชรเกษม","เพชรเกษม", "สุขุมวิท", "รามคำแหง"]
subdistrict_variants = ["บางกอกน้อย", "บางกอกใหญ่", "ธนบุรี", "บางพลัด"]
district_variants = ["คลองสาน", "ปทุมวัน","ราษฎร์บูรณะ", "ดุสิต","พญาไท", "บางรัก", "มีนบุรี"]
province_variants = selected_provinces if selected_provinces else provinces
postal_codes = ["10110", "10230" ,"20000", "10200", "10210", "10220", "10230",
    "10300", "10310""21000","30000",  # Nakhon Ratchasima
    "40000",  # Khon Kaen
    "50000",  # Chiang Mai
    "60000",  # Nakhon Sawan
    "70000",  # Ratchaburi
    "80000",  # Nakhon Si Thammarat
    "90000",  # Songkhla
    "91000",  # Satun
    "92000",  # Trang
    "93000",  # Phatthalung
    "94000",  # Pattani
    "96000",  # Narathiwat
    "97000",  ]

# Generate a random address
def generate_address(selected_formats):
    def format_component(format_key, variants):
        selected_format = random.choice(selected_formats[format_key]) if selected_formats[format_key] else ""
        return f"{selected_format}{random.choice(variants)}" if selected_format != "ไม่มี" else random.choice(variants)

    # Handle House Number directly
    house_number = random.choice(["553", "456/78", "99/1", "123หมู่1","88/22", "234", "5/3", "12", "900หมู่12", "101",
    "74/11", "87หมู่3", "55หมู่9"])

    return {
        "ชื่อ": format_component("ชื่อ", first_names),
        "บ้านเลขที่": house_number,
        "หมู่บ้าน": format_component("หมู่บ้าน", village_variants),
        "ซอย": format_component("ซอย", soi_variants),
        "ถนน": format_component("ถนน", road_variants),
        "ตำบล": format_component("ตำบล", subdistrict_variants),
        "อำเภอ": format_component("อำเภอ", district_variants),
        "จังหวัด": format_component("จังหวัด", province_variants),
        "รหัสไปรษณีย์": random.choice(postal_codes)
    }

# Generate address samples
def generate_samples(selected_formats, num_samples=50):
    sample_addresses = []
    predicted_tags_list = []
    label_list = []

    # Tag labels for components
    tag_labels = {
        "ชื่อ": "O",
        "บ้านเลขที่": "ADDR",
        "หมู่บ้าน": "ADDR",
        "ซอย": "ADDR",
        "ถนน": "ADDR",
        "ตำบล": "LOC",
        "อำเภอ": "LOC",
        "จังหวัด": "LOC",
        "รหัสไปรษณีย์": "POST"
    }

    # Define the components order and visibility
    components_order = ["ชื่อ", "บ้านเลขที่", "หมู่บ้าน", "ซอย", "ถนน", "ตำบล", "อำเภอ", "จังหวัด", "รหัสไปรษณีย์"]
    component_visibility = {component: True for component in components_order}  # Default: all visible

    for _ in range(num_samples):
        # Generate random address data
        address_data = generate_address(selected_formats)
        customized_address = " ".join([
            address_data[component]
            for component in components_order
            if component_visibility.get(component, False)
        ])

        # Collect labels based on components
        labels = [
            tag_labels[component]
            for component in components_order
            if component_visibility.get(component, False)
        ]

        # Use the `parse` function to generate predictions
        predicted_tags = parse(customized_address)  # Assume parse function returns list of tags matching words in the address

        # Append data to respective lists
        sample_addresses.append(customized_address)
        predicted_tags_list.append(predicted_tags)
        label_list.append(labels)

    return sample_addresses, predicted_tags_list, label_list

# Display samples and generate confusion matrix
# Generate and display all charts
if st.button("Generate All Charts"):
    # Generate Samples
    samples, predictions, labels = generate_samples(selected_formats)
    
    # Create DataFrame
    df_addresses = pd.DataFrame({
        "Address ": samples,
        "Predict": predictions,
        "Labels": labels,
    })
    st.dataframe(df_addresses)  # Display the DataFrame

    # Flatten Labels and Predictions for analysis
    flat_labels = [tag for sublist in df_addresses["Labels"] for tag in sublist]
    flat_predictions = [tag for sublist in df_addresses["Predict"] for tag in sublist]
    
    # Confusion Matrix
    # Confusion Matrix
    def create_confusion_matrix(true_labels, predicted_labels):
        labels = ["O", "LOC", "POST", "ADDR"]  # Define label categories
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        return pd.DataFrame(cm, index=labels, columns=labels)

    cm_df = create_confusion_matrix(flat_labels, flat_predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    #sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linecolor="white", linewidths=0.5, ax=ax)
    ax.set_xlabel("Predict", fontsize=12)
    ax.set_ylabel("Label", fontsize=12)
    #st.pyplot(fig)

    # Sentence-Level Prediction Patterns Bar Chart
    sentence_patterns = [" ".join(pred) for pred in df_addresses["Predict"]]
    pattern_counts = pd.Series(sentence_patterns).value_counts().reset_index()
    pattern_counts.columns = ['Pattern', 'Count']
    pattern_counts = pattern_counts.sort_values(by="Count", ascending=True)
    # Prediction Sentence Patterns
    st.markdown("### Prediction Sentence Patterns")
    fig = px.bar(
        pattern_counts,
        x="Count",
        y="Pattern",
        orientation='h',
        #title="Frequency of Prediction Patterns",
        labels={"Count": "Count", "Pattern": "Prediction Pattern"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Match Comparison Chart
    # Match Comparison Chart
    comparison_df = pd.DataFrame({
        "Label": flat_labels,
        "Predict": flat_predictions
    })
    comparison_df["Match"] = comparison_df["Label"] == comparison_df["Predict"]

    # Count matches and mismatches
    comparison_counts = comparison_df.groupby(["Label", "Match"]).size().reset_index(name="Count")
    comparison_pivot = comparison_counts.pivot(index="Label", columns="Match", values="Count").fillna(0)
    comparison_pivot.columns = ["Incorrect", "Correct"]  # Update column names
    comparison_pivot = comparison_pivot.reset_index()

    st.markdown("### Prediction Comparison Chart")
    comparison_melted = comparison_pivot.melt(id_vars="Label", var_name="Match Type", value_name="Count")

    # Define color scheme alternating "Correct" and "Incorrect"
    custom_colors = {
        "Correct": "#0068C9",  # Same blue color as in the original chart
        "Incorrect": "#83C9FF"  # Same orange color as in the original chart
    }

    fig = px.bar(
        comparison_melted,
        x="Label",
        y="Count",
        color="Match Type",
        color_discrete_map=custom_colors,  # Apply custom colors
        barmode="stack",
        labels={"Label": "Label", "Count": "Count", "Match Type": "Match Type"}
    )

    # Sort bars to alternate "Correct" and "Incorrect"
    fig.for_each_trace(lambda trace: trace.update(name="Correct" if trace.name == "Correct" else "Incorrect"))
    fig.data = sorted(fig.data, key=lambda x: x.name, reverse=False)  # Ensure alternating stacking

    # Plotly chart
    st.plotly_chart(fig, use_container_width=True)

    # Component-wise Correct/Incorrect Summary
    st.markdown("#### Component of Address (Correct vs Incorrect)")
    components_order = ["Name", "HouseNumber", "Village", "Soi", "Road", "Subdistrict", "District", "Province", "PostalCode"]
    component_summary = []

    # Analyze correct and incorrect counts for each component
    for i, component in enumerate(components_order):
        correct_count = 0
        incorrect_count = 0
        for pred, label in zip(predictions, labels):
            if i < len(pred) and i < len(label):  # Avoid index errors
                if pred[i] == label[i]:
                    correct_count += 1
                else:
                    incorrect_count += 1
        component_summary.append({"Component": component, "Correct": correct_count, "Incorrect": incorrect_count})

    # Convert to DataFrame for visualization
    summary_df = pd.DataFrame(component_summary)

    # Create an interactive bar chart using Plotly
    fig = px.bar(
        summary_df.melt(id_vars="Component", var_name="Type", value_name="Count"),
        x="Component",
        y="Count",
        color="Type",
        barmode="stack",
        #title="Component-Wise Correct vs Incorrect Count",
        labels={"Component": "Component", "Count": "Count", "Type": "Type"}
    )

    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)

    # Display the summary table
    st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)
