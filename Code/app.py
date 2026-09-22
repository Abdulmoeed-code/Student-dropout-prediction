import gradio as gr
import pandas as pd
import joblib


joblib.dump(dropout_model, 'student_dropout_model.pkl')
joblib.dump(data_preprocessor, 'data_preprocessor.pkl')


model = joblib.load('student_dropout_model.pkl')
preprocessor = joblib.load('data_preprocessor.pkl')


def predict_dropout(Age, Gender, Family_Income, Internet_Access, Study_Hours_per_Day,
                    Attendance_Rate, Assignment_Delay_Days, Travel_Time_Minutes,
                    Part_Time_Job, Scholarship, Stress_Index, GPA, Semester_GPA,
                    CGPA, Semester, Department, Parental_Education):


    input_data = pd.DataFrame([[Age, Gender, Family_Income, Internet_Access, Study_Hours_per_Day,
                                Attendance_Rate, Assignment_Delay_Days, Travel_Time_Minutes,
                                Part_Time_Job, Scholarship, Stress_Index, GPA, Semester_GPA,
                                CGPA, Semester, Department, Parental_Education]],
                              columns=['Age', 'Gender', 'Family_Income', 'Internet_Access', 'Study_Hours_per_Day',
                                       'Attendance_Rate', 'Assignment_Delay_Days', 'Travel_Time_Minutes',
                                       'Part_Time_Job', 'Scholarship', 'Stress_Index', 'GPA', 'Semester_GPA',
                                       'CGPA', 'Semester', 'Department', 'Parental_Education'])


    processed_data = preprocessor.transform(input_data)
    probability = model.predict_proba(processed_data)[0][1]

   
    if probability >= 0.70:
        risk = "High Risk"
    elif probability >= 0.40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return round(probability, 4), risk


inputs = [
    gr.Number(label="Age"),
    gr.Dropdown(['Male', 'Female'], label="Gender"),
    gr.Number(label="Family Income"),
    gr.Dropdown(['Yes', 'No'], label="Internet Access"),
    gr.Number(label="Study Hours per Day"),
    gr.Number(label="Attendance Rate"),
    gr.Number(label="Assignment Delay Days"),
    gr.Number(label="Travel Time Minutes"),
    gr.Dropdown(['Yes', 'No'], label="Part Time Job"),
    gr.Dropdown(['Yes', 'No'], label="Scholarship"),
    gr.Number(label="Stress Index"),
    gr.Number(label="GPA"),
    gr.Number(label="Semester GPA"),
    gr.Number(label="CGPA"),
    gr.Dropdown(['Year 1', 'Year 2', 'Year 3', 'Year 4'], label="Semester"),
    gr.Dropdown(['Arts', 'Business', 'CS', 'Engineering', 'Science'], label="Department"),
    gr.Dropdown(['High School', 'Bachelor', 'Master', 'PhD'], label="Parental Education")
]

outputs = [
    gr.Textbox(label="Probability Result (0 to 1)"),
    gr.Textbox(label="Risk Category")
]


app = gr.Interface(fn=predict_dropout, inputs=inputs, outputs=outputs, title="Student Dropout Predictor")
app.launch(share=True)