"""
Generates data/sample_medquad.csv — a small, hand-written set of Q&A pairs in the
same *shape* as MedQuAD (question, answer, focus disease, question_type, source)
so the chatbot works out-of-the-box without needing to download the full dataset.

This is NOT a copy of MedQuAD content. It exists only so the app has something to
retrieve from during a demo. For real use, run build_dataset.py against a local
clone of https://github.com/abachaa/MedQuAD to build the full dataset.
"""
import pandas as pd
import os

ROWS = [
    ("diabetes", "What is diabetes?", "Diabetes is a chronic condition in which the body cannot properly regulate blood sugar (glucose) levels, either because the pancreas does not produce enough insulin (Type 1) or because the body's cells do not respond well to insulin (Type 2).", "information", "NIDDK"),
    ("diabetes", "What are the symptoms of diabetes?", "Common symptoms include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, and slow-healing sores.", "symptoms", "NIDDK"),
    ("diabetes", "How is diabetes treated?", "Treatment usually includes blood sugar monitoring, insulin or oral medication, dietary changes, regular exercise, and routine check-ups to manage complications.", "treatment", "NIDDK"),
    ("hypertension", "What is high blood pressure?", "High blood pressure, or hypertension, occurs when the force of blood pushing against artery walls is consistently too high, which can damage the heart and blood vessels over time.", "information", "NHLBI"),
    ("hypertension", "What causes high blood pressure?", "Causes include a diet high in salt, lack of physical activity, obesity, smoking, excessive alcohol use, stress, and genetic factors.", "causes", "NHLBI"),
    ("hypertension", "How can I lower my blood pressure?", "Lifestyle changes such as reducing salt intake, exercising regularly, maintaining a healthy weight, limiting alcohol, and managing stress can help. Medication such as ACE inhibitors, diuretics, or beta blockers may also be prescribed.", "treatment", "NHLBI"),
    ("asthma", "What is asthma?", "Asthma is a chronic lung disease in which the airways become inflamed and narrowed, causing wheezing, coughing, chest tightness, and shortness of breath.", "information", "NHLBI"),
    ("asthma", "What triggers asthma attacks?", "Common triggers include allergens like pollen and dust mites, respiratory infections, cold air, exercise, smoke, and strong odors.", "causes", "NHLBI"),
    ("asthma", "How is asthma treated?", "Asthma is managed with quick-relief inhalers for sudden symptoms and long-term control medications such as inhaled corticosteroids to reduce airway inflammation.", "treatment", "NHLBI"),
    ("migraine", "What is a migraine?", "A migraine is a neurological condition causing intense, throbbing headaches, often on one side of the head, sometimes accompanied by nausea, vomiting, and sensitivity to light and sound.", "information", "NINDS"),
    ("migraine", "What causes migraines?", "Migraines can be triggered by hormonal changes, certain foods, stress, lack of sleep, bright lights, and changes in weather, though the exact cause is not fully understood.", "causes", "NINDS"),
    ("common cold", "What is the common cold?", "The common cold is a mild viral infection of the nose and throat, usually caused by rhinoviruses, leading to symptoms like a runny nose, sneezing, and sore throat.", "information", "NIAID"),
    ("common cold", "How is the common cold treated?", "There is no cure for the common cold. Treatment focuses on rest, fluids, and over-the-counter medication to relieve symptoms such as congestion and sore throat.", "treatment", "NIAID"),
    ("covid-19", "What is COVID-19?", "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus, which can cause symptoms ranging from mild cold-like symptoms to severe pneumonia and breathing difficulty.", "information", "CDC"),
    ("covid-19", "What are the symptoms of COVID-19?", "Symptoms may include fever, cough, fatigue, loss of taste or smell, sore throat, shortness of breath, and body aches.", "symptoms", "CDC"),
    ("covid-19", "How can COVID-19 be prevented?", "Prevention includes vaccination, frequent handwashing, wearing a mask in crowded indoor settings, and staying home when sick.", "prevention", "CDC"),
    ("pneumonia", "What is pneumonia?", "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid, causing cough, fever, chills, and difficulty breathing.", "information", "NHLBI"),
    ("pneumonia", "How is pneumonia treated?", "Bacterial pneumonia is treated with antibiotics, while viral pneumonia is managed with rest, fluids, and medication to control fever. Severe cases may require hospitalization.", "treatment", "NHLBI"),
    ("arthritis", "What is arthritis?", "Arthritis refers to inflammation of one or more joints, causing pain and stiffness. Osteoarthritis and rheumatoid arthritis are among the most common types.", "information", "NIAMS"),
    ("arthritis", "How is arthritis treated?", "Treatment may include pain relievers, anti-inflammatory medication, physical therapy, exercise, weight management, and in severe cases, joint replacement surgery.", "treatment", "NIAMS"),
    ("depression", "What is depression?", "Depression is a mood disorder that causes persistent feelings of sadness, hopelessness, and loss of interest in activities, which can affect daily functioning.", "information", "NIMH"),
    ("depression", "What are the symptoms of depression?", "Symptoms include persistent sadness, fatigue, changes in appetite or sleep, difficulty concentrating, feelings of worthlessness, and in severe cases, thoughts of self-harm.", "symptoms", "NIMH"),
    ("depression", "How is depression treated?", "Depression is commonly treated with psychotherapy such as cognitive behavioral therapy, antidepressant medication, lifestyle changes, and in some cases a combination of these approaches.", "treatment", "NIMH"),
    ("gerd", "What is GERD?", "Gastroesophageal reflux disease (GERD) occurs when stomach acid frequently flows back into the esophagus, irritating its lining and causing heartburn and regurgitation.", "information", "NIDDK"),
    ("gerd", "How is GERD treated?", "Treatment may include lifestyle changes such as avoiding trigger foods, weight loss, and medications like antacids, H2 blockers, or proton pump inhibitors.", "treatment", "NIDDK"),
    ("heart disease", "What is coronary artery disease?", "Coronary artery disease occurs when the blood vessels supplying the heart become narrowed or blocked by fatty deposits, increasing the risk of chest pain and heart attack.", "information", "NHLBI"),
    ("heart disease", "What are the symptoms of a heart attack?", "Symptoms include chest pain or pressure, pain spreading to the arm or jaw, shortness of breath, cold sweat, nausea, and lightheadedness. Seek emergency care immediately if these occur.", "symptoms", "NHLBI"),
    ("stroke", "What is a stroke?", "A stroke occurs when blood flow to part of the brain is interrupted or reduced, depriving brain tissue of oxygen and nutrients, which can cause brain cells to die within minutes.", "information", "NINDS"),
    ("stroke", "What are the warning signs of a stroke?", "Warning signs include sudden numbness or weakness on one side of the body, confusion, trouble speaking, vision problems, and difficulty walking. Remember the acronym FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services.", "symptoms", "NINDS"),
    ("obesity", "What is obesity?", "Obesity is a condition involving excess body fat that increases the risk of health problems such as heart disease, diabetes, and high blood pressure.", "information", "NIDDK"),
    ("obesity", "How is obesity managed?", "Management typically involves dietary changes, increased physical activity, behavioral therapy, and in some cases, medication or weight-loss surgery.", "treatment", "NIDDK"),
    ("anemia", "What is anemia?", "Anemia is a condition in which the blood lacks enough healthy red blood cells or hemoglobin to carry adequate oxygen to the body's tissues, causing fatigue and weakness.", "information", "NHLBI"),
    ("hepatitis", "What is hepatitis?", "Hepatitis is inflammation of the liver, most often caused by viral infection, but also by alcohol use, toxins, or autoimmune disease.", "information", "NIDDK"),
    ("tuberculosis", "What is tuberculosis?", "Tuberculosis is a bacterial infection that mainly affects the lungs, spreading through the air when an infected person coughs or sneezes, and causing symptoms like persistent cough and weight loss.", "information", "CDC"),
    ("chronic kidney disease", "What is chronic kidney disease?", "Chronic kidney disease is the gradual loss of kidney function over time, which can lead to a buildup of waste in the body and eventually kidney failure if untreated.", "information", "NIDDK"),
    ("crohn's disease", "What is Crohn's disease?", "Crohn's disease is a type of inflammatory bowel disease that causes chronic inflammation of the digestive tract, leading to abdominal pain, diarrhea, fatigue, and weight loss.", "information", "NIDDK"),
    ("epilepsy", "What is epilepsy?", "Epilepsy is a neurological disorder marked by recurrent, unprovoked seizures caused by abnormal electrical activity in the brain.", "information", "NINDS"),
    ("alzheimer's disease", "What is Alzheimer's disease?", "Alzheimer's disease is a progressive brain disorder that slowly destroys memory and thinking skills, eventually affecting the ability to carry out simple tasks.", "information", "NIA"),
    ("alzheimer's disease", "What are the early signs of Alzheimer's disease?", "Early signs include memory loss that disrupts daily life, difficulty planning or solving problems, confusion with time or place, and withdrawal from work or social activities.", "symptoms", "NIA"),
    ("lupus", "What is lupus?", "Lupus is a chronic autoimmune disease in which the immune system attacks the body's own tissues, potentially affecting the skin, joints, kidneys, and other organs.", "information", "NIAMS"),
    ("psoriasis", "What is psoriasis?", "Psoriasis is a chronic skin condition that causes cells to build up rapidly on the skin's surface, forming scaly, itchy, red patches.", "information", "NIAMS"),
    ("allergies", "What are allergies?", "Allergies occur when the immune system reacts abnormally to a normally harmless substance, called an allergen, causing symptoms such as sneezing, itching, or hives.", "information", "NIAID"),
    ("urinary tract infection", "What is a urinary tract infection?", "A urinary tract infection (UTI) is an infection in any part of the urinary system, most commonly the bladder, causing symptoms like burning during urination and frequent urges to urinate.", "information", "NIDDK"),
    ("thyroid disease", "What is hypothyroidism?", "Hypothyroidism is a condition in which the thyroid gland does not produce enough thyroid hormone, which can slow metabolism and cause fatigue, weight gain, and cold sensitivity.", "information", "NIDDK"),
    ("gout", "What is gout?", "Gout is a form of arthritis caused by a buildup of uric acid crystals in the joints, leading to sudden, severe pain, redness, and swelling, often in the big toe.", "information", "NIAMS"),
    ("sleep apnea", "What is sleep apnea?", "Sleep apnea is a sleep disorder in which breathing repeatedly stops and starts during sleep, often due to blocked airways, leading to poor sleep quality and daytime fatigue.", "information", "NHLBI"),
    ("copd", "What is COPD?", "Chronic obstructive pulmonary disease (COPD) is a group of progressive lung diseases, including emphysema and chronic bronchitis, that cause airflow blockage and breathing difficulties.", "information", "NHLBI"),
    ("meningitis", "What is meningitis?", "Meningitis is inflammation of the membranes surrounding the brain and spinal cord, usually caused by a bacterial or viral infection, and can be life-threatening if not treated promptly.", "information", "CDC"),
    ("glaucoma", "What is glaucoma?", "Glaucoma is a group of eye conditions that damage the optic nerve, often due to abnormally high pressure in the eye, and can lead to vision loss if untreated.", "information", "NEI"),
    ("endometriosis", "What is endometriosis?", "Endometriosis is a condition in which tissue similar to the lining of the uterus grows outside the uterus, causing pelvic pain and sometimes fertility problems.", "information", "NICHD"),
    ("menopause", "What is menopause?", "Menopause is the natural biological process marking the end of a woman's menstrual cycles, typically diagnosed after 12 months without a period, often accompanied by hot flashes and mood changes.", "information", "NIA"),
]

def main():
    df = pd.DataFrame(ROWS, columns=["focus", "question", "answer", "question_type", "source"])
    df.insert(0, "qid", [f"sample-{i+1:04d}" for i in range(len(df))])
    out_path = os.path.join(os.path.dirname(__file__), "sample_medquad.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
