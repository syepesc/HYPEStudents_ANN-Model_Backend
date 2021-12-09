import pickle
import pandas as pd
import tensorflow as tf
from flask import Flask, request
from flask_cors import cross_origin

"""
LOAD MODEL
"""
file = open('./model/encoder', 'rb')
deserializer = pickle.load(file)
label_encoders = deserializer['label encoders']
pca = deserializer['pca']
file.close()

model = tf.keras.models.load_model('./model/deep and cross network')


"""
SERVER CONFIG
"""
app = Flask(__name__) # create app instance


"""
ROUTES
"""
@app.route('/', methods=['GET'])
@cross_origin()
def home():
	print('REACHED HOME PAGE')
	return 'HOME PAGE'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
	print('REACHED PREDICT PAGE')
	# 1) grab POST request data
	form = dict(request.json)

	# 2) define form keys as dataset columns for matching model input columns
	# organized in the same order of the columns in csv dataset file
	form_column_dict = {
	'intakeTermCode': 'INTAKE TERM CODE',
	'admitTermCode': 'ADMIT TERM CODE',
	'intakeCollegeExperience': 'INTAKE COLLEGE EXPERIENCE',
	'primaryProgramCode': 'PRIMARY PROGRAM CODE',
	'schoolCode': 'SCHOOL CODE',
	'studentLevelName': 'STUDENT LEVEL NAME',
	'mailingPostalCodeGroup3': 'MAILING POSTAL CODE GROUP 3',
	'gender': 'GENDER',
	'disabilityInd': 'DISABILITY IND',
	'futureTermEnroll': 'FUTURE TERM ENROL',
	'academicPerformance': 'ACADEMIC PERFORMANCE',
	'expectedGradTermCode': 'EXPECTED GRAD TERM CODE',
	'firstYearPersistenceCount': 'FIRST YEAR PERSISTENCE COUNT',
	'hsAverageMarks': 'HS AVERAGE MARKS',
	'englishTestScore': 'ENGLISH TEST SCORE',
	'ageGroupLongName': 'AGE GROUP LONG NAME',
	'firstGenerationInd': 'FIRST GENERATION IND',
	'effectiveAcademicHistory': 'effective academic history',
	'applicantTargetSegmentName': 'APPLICANT TARGET SEGMENT NAME',
	'timeStatusName': 'TIME STATUS NAME',
	'fundingSourceName': 'FUNDING SOURCE NAME',
	}

	# 3) format data into model input dataframe format
	sample = {}
	print("\nFORM KEYS, VALUES and TYPES")
	print("-----------------------------")
	for key in form_column_dict.keys():
		print(key, ":>>", form.get(key), ":>>", type(form.get(key)))
		sample[form_column_dict.get(key)] = form.get(key)
	# for key, value in form.items():
	# 	print(key, ":>>", value, ":>>", type(value))
	# 	sample[form_column_dict.get(key)] = value
	print("\n")

	# 4) final format
	sample = pd.DataFrame([sample])
	print("FORMATED FORM (ready to transform into a DF)")
	print("--------------------------------------------")
	print(type(sample))
	print(sample)
	print("\n")

	# 5) encoding categorical features
	categorical_column = [col for col in sample.columns if sample[col].dtype == 'object'] + ['PRIMARY PROGRAM CODE', 'INTAKE TERM CODE', 'ADMIT TERM CODE', 'EXPECTED GRAD TERM CODE']
	for column in categorical_column:
		encoder = label_encoders[column]
		print(column + ':')
		print(encoder.classes_)
		sample[column] = pd.Series(encoder.transform(sample[column][sample[column].notna()]), index=sample[column][sample[column].notna()].index)
	# 6) feature reduction
	reduced_feature = pca.transform(sample[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])  # FUNDING SOURCE NAME is highly correlated with TIME STATUS NAME
	sample.insert(6, 'time and fund', reduced_feature)
	sample.drop(columns=['FUNDING SOURCE NAME', 'TIME STATUS NAME'], inplace=True)
	prediction = model.predict(sample)[:, 1][0]

	if prediction < .5:
		return {"prediction":str(prediction), "graphs":[]} 
	else:
		return {"prediction":str(prediction), "graphs":["academic performance vs failure.png", "first year survival vs failure.png", "high school marks vs failure.png"]} 


# Start Flash Server
# Since there is no main() function in Python,
# when the command to run a python program is given to the interpreter,
# the code that is at level 0 indentation is to be executed.
# However, before doing that, it will define a few special variables.
# __name__ is one such special variable. If the source file is executed as the main program,
# the interpreter sets the __name__ variable to have a value “__main__�?
# If this file is being imported from another module, __name__ will be set to the module’s name.
if __name__ == '__main__':
	app.run(debug=True) # turn off if production mode













########
# TEST #
########

# sample = {
# 	'INTAKE TERM CODE': 2020, 
# 	'ADMIT TERM CODE': 2020, 
# 	'INTAKE COLLEGE EXPERIENCE': 'New to College', 
# 	'PRIMARY PROGRAM CODE': 6700, 
# 	'SCHOOL CODE': 'CH', 
# 	'STUDENT LEVEL NAME': 'Post Secondary', 
# 	'MAILING POSTAL CODE GROUP 3': 'L1V', 
# 	'GENDER': 'F', 
# 	'DISABILITY IND': 'N', 
# 	'FUTURE TERM ENROL': '1-0-0-0-0-0-0-0-0-0',
# 	'ACADEMIC PERFORMANCE': 'DF - Poor', 
# 	'EXPECTED GRAD TERM CODE': 2022, 
# 	'FIRST YEAR PERSISTENCE COUNT': 0, 
# 	'HS AVERAGE MARKS': 0, 
# 	'ENGLISH TEST SCORE': 200, 
# 	'AGE GROUP LONG NAME': '21 to 25', 
# 	'FIRST GENERATION IND': 'N', 
# 	'effective academic history': 'no', 
# 	'APPLICANT TARGET SEGMENT NAME': 'Non-Direct Entry',
# 	'TIME STATUS NAME': 'Full-Time', 
# 	'FUNDING SOURCE NAME': 'GPOG - FT'
# 	}

# sample = {
# 	'INTAKE TERM CODE': 2020, 
# 	'ADMIT TERM CODE': 2020, 
# 	'INTAKE COLLEGE EXPERIENCE': 'New to College', 
# 	'PRIMARY PROGRAM CODE': 9111, 
# 	'SCHOOL CODE': 'CH', 
# 	'STUDENT LEVEL NAME': 'Post Secondary', 
# 	'MAILING POSTAL CODE GROUP 3': 'L1V', 
# 	'GENDER': 'F', 
# 	'DISABILITY IND': 'N', 
# 	'FUTURE TERM ENROL': '1-1-1-1-0-0-0-0-0-0',
# 	'ACADEMIC PERFORMANCE': 'DF - Poor', 
# 	'EXPECTED GRAD TERM CODE': 2020, 
# 	'FIRST YEAR PERSISTENCE COUNT': 0, 
# 	'HS AVERAGE MARKS': 0, 
# 	'ENGLISH TEST SCORE': 140, 
# 	'AGE GROUP LONG NAME': '41 to 50', 
# 	'FIRST GENERATION IND': 'Y', 
# 	'effective academic history': 'no', 
# 	'APPLICANT TARGET SEGMENT NAME': 'Non-Direct Entry',
# 	'TIME STATUS NAME': 'Full-Time', 
# 	'FUNDING SOURCE NAME': 'GPOG - FT'
# 	}


# sample = pd.DataFrame([sample])
# print("FORMATED FORM (ready to transform into a DF)")
# print("--------------------------------------------")
# print(type(sample))
# print(sample)
# print("\n")

# # 5) encoding categorical features
# categorical_column = [col for col in sample.columns if sample[col].dtype == 'object'] + ['PRIMARY PROGRAM CODE', 'INTAKE TERM CODE', 'ADMIT TERM CODE', 'EXPECTED GRAD TERM CODE']
# for column in categorical_column:
# 	encoder = label_encoders[column]
# 	print(column + ':')
# 	print(encoder.classes_)
# 	sample[column] = pd.Series(encoder.transform(sample[column][sample[column].notna()]), index=sample[column][sample[column].notna()].index)
# # 6) feature reduction
# reduced_feature = pca.transform(sample[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])  # FUNDING SOURCE NAME is highly correlated with TIME STATUS NAME
# sample.insert(6, 'time and fund', reduced_feature)
# sample.drop(columns=['FUNDING SOURCE NAME', 'TIME STATUS NAME'], inplace=True)
# prediction = model.predict(sample)[:, 1][0]

# if prediction < .5:
# 	print({"prediction":prediction, "graphs":["academic performance vs failure.png", "first year survival vs failure.png", "high school marks vs failure.png"]}) # '%.2f % to fail. well done! keep hard working'
# else:
# 	print({"prediction":prediction, "graphs":[]}) # '%.2f % to fail.'