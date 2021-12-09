import pandas as pd, pickle, tensorflow as tf

sample = [{'INTAKE TERM CODE': 2021, 'ADMIT TERM CODE': 2021, 'INTAKE COLLEGE EXPERIENCE': 'Enrolled', 'PRIMARY PROGRAM CODE': 8226, 'SCHOOL CODE': 'TR', 'STUDENT LEVEL NAME': 'Post Secondary', 'TIME STATUS NAME': 'Full-Time', 'FUNDING SOURCE NAME': 'GPOG - FT', 'MAILING POSTAL CODE GROUP 3': 'M4H', 'GENDER': 'M', 'DISABILITY IND': 'N', 'FUTURE TERM ENROL': '1-0-0-0-0-0-0-0-0-0', 'ACADEMIC PERFORMANCE': 'C - Satisfactory', 'EXPECTED GRAD TERM CODE': 2023, 'FIRST YEAR PERSISTENCE COUNT': 0, 'HS AVERAGE MARKS': 75, 'ENGLISH TEST SCORE': 130, 'AGE GROUP LONG NAME': '0 to 18', 'FIRST GENERATION IND': 'N', 'effective academic history': 'high school', 'APPLICANT TARGET SEGMENT NAME': 'Direct Entry'}]
sample = pd.DataFrame(sample)
file = open('encoder', 'rb')
deserializer = pickle.load(file)
label_encoders = deserializer['label encoders']
pca = deserializer['pca']
file.close()

# encoding categorical features
categorical_column = [col for col in sample.columns if sample[col].dtype == 'object']
for column in categorical_column:
	encoder = label_encoders[column]
	print(column + ':')
	print(encoder.classes_)
	sample[column] = pd.Series(encoder.fit_transform(sample[column][sample[column].notna()]), index=sample[column][sample[column].notna()].index)
# feature reduction
reduced_feature = pca.transform(sample[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])  # FUNDING SOURCE NAME is highly correlated with TIME STATUS NAME
sample.insert(6, 'time and fund', reduced_feature)
sample.drop(columns=['FUNDING SOURCE NAME', 'TIME STATUS NAME'], inplace=True)

model = tf.keras.models.load_model('deep and cross network')
prediction = model.predict(sample)[:, 1][0]
print(prediction)