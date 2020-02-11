import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Select, Dropdown, CheckboxGroup, Div, FixedTicker, Panel, Tabs, Button
from bokeh.plotting import figure, output_file, show
from bokeh.models.glyphs import Rect

import matplotlib
matplotlib.use('Agg')
import healthForecaster as hF
import pickle 
# pickle.dump([trainedModels, inputFeatures, responseVariables], open("models.p", "wb"))
[trainedModels, inputFeatures, responseVariables, data] = pickle.load( open( "models.p", "rb" ) )
# Alternatively extracts, parses, and trains the models:
# [trainedModels, inputFeatures, responseVariables] = hF.createHealthForecasterModels()

units = ['uIU/mL', 'mmol/L', 'mmol/L', 'mmol/L', 'mmol/L', 'mg/dL',
 'g/L', 'mg/dL', 'mg/dL', 'μMol/L', 'mg/dL', 'mmol/L', 'ng/ML',
 'g/L', '*10^9', '*10^9', '*10^9', 'mmol/L', 'mmol/L', 'U/L', 'g/L', 'g/L',
 'mmol/L', 'g/L', 'g/L', 'mg/L', 'mmHG', 'mmHG']
unitDict = dict(zip(responseVariables, units))
unitDictVals = list(unitDict.values())


# Define initial settings: 
inputDict = dict(modelName = 'randomForest', Age = 65, Sex = 'M',
											 Location = 'urban',
											 Medical_condition = 'noReport', Pregnant = 0, Diabetes = 0, High_BP = 0, Heart_attack = 0, Internal_bleeding = 0,
											 Height = 177.8, currWeight = 90.7029, 
											 currSmoker = 0, intSmoker = 0,
											 currCups_water_daily= 3,
											 currAlcohol_frequency=2, 
											 currDailyScreenTime = 60, 
											 currHours_of_sleep = 8, 
											 currCarbo = 500, 
											 currFat = 100, 
											 currProtn = 100, 
											 currActivityLevel = 1) 

# set intervention values to current values
inputDict.update(dict(intWeight = inputDict['currWeight'], intCups_water_daily= inputDict['currCups_water_daily'], intAlcohol_frequency=inputDict['currAlcohol_frequency'],
	intSmoker = inputDict['currSmoker'], intDailyScreenTime = inputDict['currDailyScreenTime'], intHours_of_sleep = inputDict['currHours_of_sleep'],
	intCarbo = inputDict['currCarbo'], intFat = inputDict['currFat'], intProtn = inputDict['currProtn'], intActivityLevel = inputDict['currActivityLevel']))

medianBlood = np.round(data[responseVariables].median().to_numpy(), decimals=2)
medianBlood = medianBlood[:-2]
blood_std = np.round(data[responseVariables].std().to_numpy(), decimals = 2)
blood_std = blood_std[:-2]


# initialBlood = [ 16.86,   1.92, 1.54,   3.22,   5.6,  274.54,   1.19, 508.12,   2.59,  82.42,
#     0.99,   0.93, 138.07, 130.61,   5.99,   4.42, 208.21,   5.19,   5.64,  77.76,
#    46.86,   5.27,   5.19,  20.11, 271.82,   1.41]

initialBlood = [21.02,   2.14,   1.23,   3.12,  5.97, 383.56,   1.1,  128.02,   3.2,   98.24,
1.,     0.96, 202.75, 155.83,   6.64,   5.06, 194.27,   5.95,   5.98,  76.84,
47.77,   5.97,   4.99,  27.33, 290.63,   1.41]

############ Define widgets
def createCurrBloodInputFields():
	insText        =    TextInput(title="Insulin(uIU/mL)                       ", value=str(initialBlood[0]   ))
	tgText         =    TextInput(title="Triglycerides(mmol/L)                 ", value=str(initialBlood[1]   ))
	hdl_cText      =    TextInput(title="HDL_C(mmol/L)                         ", value=str(initialBlood[2]   ))
	ldl_cText      =    TextInput(title="LDL_C(mmol/L)                         ", value=str(initialBlood[3]   ))
	ureaText       =    TextInput(title="Urea(mmol/L)                          ", value=str(initialBlood[4]   ))
	uaText         =    TextInput(title="Uric_acid(mg/dL)                      ", value=str(initialBlood[5]   ))
	apo_aText      =    TextInput(title="APO-A(g/L)                            ", value=str(initialBlood[6]   ))
	lp_aText       =    TextInput(title="Lipoprotein-A (mg/dL)                  ", value=str(initialBlood[7]   ))
	hs_crpText     =    TextInput(title="High-sensitivity_CRP(mg/dL)           ", value=str(initialBlood[8]   ))
	creText        =    TextInput(title="Creatinine(μMol/L)                    ", value=str(initialBlood[9]   ))
	apo_bText      =    TextInput(title="APO-B(mg/dL)                          ", value=str(initialBlood[10]  ))
	mgText         =    TextInput(title="Magnesium(mmol/L)                     ", value=str(initialBlood[11]  ))
	fetText        =    TextInput(title="Ferritin(ng/ML)                       ", value=str(initialBlood[12]  ))
	hgbText        =    TextInput(title="Hemoglobin(g/L)                       ", value=str(initialBlood[13]  ))
	wbcText        =    TextInput(title="White_blood_cell_count(10^9)          ", value=str(initialBlood[14]  ))
	rbcText        =    TextInput(title="Red_blood_cell_count(10^9)            ", value=str(initialBlood[15]  ))
	pltText        =    TextInput(title="Platelet_count(10^9)                  ", value=str(initialBlood[16]  ))
	glu_fieldText  =    TextInput(title="Glucose_field(mmol/L)                 ", value=str(initialBlood[17]  ))
	hba1cText      =    TextInput(title="HbA1c(mmol/L)                         ", value=str(initialBlood[18]  ))
	tpText         =    TextInput(title="Total_protein(g/L)                    ", value=str(initialBlood[19]  ))
	albText        =    TextInput(title="Albumin(g/L)                          ", value=str(initialBlood[20]  ))
	glucoseText    =    TextInput(title="Glucose(mmol/L)                       ", value=str(initialBlood[21]  ))
	tcText         =    TextInput(title="Total_cholesterol(g/L)                ", value=str(initialBlood[22]  ))
	altText        =    TextInput(title="Alanine_aminotransferase(U/L)         ", value=str(initialBlood[23]  ))
	trfText        =    TextInput(title="Transferrin(g/L)                      ", value=str(initialBlood[24]  ))
	trf_rText      =    TextInput(title="Transferrin_receptor(mg/L)            ", value=str(initialBlood[25]  ))

	clearButton = Button(label = 'Clear my test')

	currBloodInputFields = [insText, tgText, hdl_cText, ldl_cText, ureaText, uaText, apo_aText,
	 lp_aText, hs_crpText, creText, apo_bText, mgText, fetText, hgbText, wbcText, rbcText, pltText,
	 glu_fieldText, hba1cText, tpText, albText, glucoseText, tcText, altText, trfText, trf_rText]

	r = 0
	tmp0 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 1
	tmp1 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 2
	tmp2 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 3
	tmp3 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 4
	tmp4 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 5
	tmp5 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 6
	tmp6 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 7
	tmp7 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], currBloodInputFields[r*3+2])
	r = 8
	tmp8 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], clearButton)
	currBloodInputLayout = column(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8)
	
	return currBloodInputFields, currBloodInputLayout, insText, tgText, ureaText, uaText, apo_aText, lp_aText, hs_crpText, creText, hdl_cText, ldl_cText, apo_bText, mgText, fetText, hgbText, wbcText, rbcText, pltText, glu_fieldText, hba1cText, tpText, albText, glucoseText, tcText, altText, trfText, trf_rText, clearButton

[currBloodInputFields, currBloodInputLayout, insText, tgText, ureaText, uaText, apo_aText, lp_aText, hs_crpText, creText, hdl_cText,
ldl_cText, apo_bText, mgText, fetText, hgbText, wbcText, rbcText, pltText, glu_fieldText,
hba1cText, tpText, albText, glucoseText, tcText, altText, trfText, trf_rText, clearButton] = createCurrBloodInputFields()

def createFixedInputs_andBP():
	modelList = ["randomForest", "ols", "ridge","elastic"]
	modelSelector = Select(title="Model", options=modelList, value = 'randomForest')
	ageSlider = Slider(title="Age", value=inputDict['Age'], start=0, end=100, step=1)
	sexSelector = Select(title="Sex", options=['M', 'F'], value = inputDict['Sex'])
	heightSlider=Slider(title="Height (inches)", value=inputDict['Height']/2.54, start=0, end=90, step=1)
	locationSelector = Select(title="Location", options=['urban', 'rural'], value = 'urban')
	
	medicalConditionList = ['noReport', 'ENT', 'OBGYN',
				 'Old_age_midLife_syndrome','alcohol_poisoning', 'dermatological',
				 'digestive','endocrine','heart','hematological',
				 'infectious_parasitic','injury','muscular_rheumatological',
				 'neurological','noDiagnosis','other','pyschiatric',
				 'respiratory','sexualDysfunction','tumor','unknown','urinary']
	medicalConditionSelector = Select(title="Medical Condition", options=medicalConditionList, value = 'noReport')

	pregnantCheckbox = CheckboxGroup(labels=["Pregnant"], active=[])
	diabetesCheckbox = CheckboxGroup(labels=["Diabetes"], active=[])
	highBPCheckbox = CheckboxGroup(labels=["High blood pressure"], active=[])
	heart_attackCheckbox = CheckboxGroup(labels=["History of Heart attack"], active=[])
	internal_bleedingCheckbox = CheckboxGroup(labels=["History of internal bleeding"], active=[])

	systolText        =    TextInput(title="Systolic_BP(mmHg)                ", value=str(130))
	diastolText        =    TextInput(title="Diastolic_BP(mmHg)               ", value=str(80))

	fixedInputs_andBP_layout = column(modelSelector, Div(text='Background:',width=300, style={'font-size': '200%', 'font-weight':'bold', 'color': 'black'}), 
	ageSlider, sexSelector, heightSlider, locationSelector, medicalConditionSelector)#,
	# pregnantCheckbox, diabetesCheckbox, highBPCheckbox, heart_attackCheckbox, internal_bleedingCheckbox, systolText, diastolText)
	
	return fixedInputs_andBP_layout, modelSelector, ageSlider, sexSelector, heightSlider, locationSelector, medicalConditionSelector, pregnantCheckbox, diabetesCheckbox, highBPCheckbox, heart_attackCheckbox, internal_bleedingCheckbox, systolText, diastolText

def createMaleableInputs():
	currWeightSlider = Slider(title="Base weight", value=inputDict['currWeight']*2.205, start=0, end=400, step=1) 
	intWeightSlider = Slider(title="Int weight", value=inputDict['intWeight']*2.205, start=0, end=400, step=1) 
	currSmokerCheckbox = CheckboxGroup(labels=["Base smoking"], active=[])
	intSmokerCheckbox = CheckboxGroup(labels=["Int smoking"], active=[])
	currWaterSlider = Slider(title="Base cups of water", value=3, start=0, end=20, step=1) 
	intWaterSlider = Slider(title="Int cups of water", value=3, start=0, end=20, step=1) 

	alcoholdFrequencyList = ['daily','3-4 times a week',
	'Once or twice a week', 'Once or twice a month', 
	'No more than once a month']
	currAlcohol_freqeuncySelector = Select(title="Base alcohol frequency", options=alcoholdFrequencyList, value = '3-4 times a week')
	intAlcohol_freqeuncySelector = Select(title="Int alcohol frequency", options = alcoholdFrequencyList, value = '3-4 times a week')

	currDailyScreenTimeSlider = Slider(title="Base screen time", value=inputDict['currDailyScreenTime'], start=0, end=300, step=10)
	intDailyScreenTimeSlider = Slider(title="Int screen time", value=inputDict['intDailyScreenTime'], start=0, end=300, step=10)
	currActivityLevelSlider = Slider(title="Base activity level", value=2, start=1, end=5, step=1)
	intActivityLevelSlider = Slider(title="Int activity", value=2, start=1, end=5, step=1)
	currHours_of_sleepSlider = Slider(title="Base sleep (hrs)", value=7, start=1, end=14, step=1)
	intHours_of_sleepSlider = Slider(title="Int sleep (hrs)", value=7, start=1, end=14, step=1)

	currCarboSlider = Slider(title="Base carbs (g)", value=inputDict['currCarbo'], start=0, end=600, step=20)
	intCarboSlider = Slider(title="Int carbs (g)", value=inputDict['intCarbo'], start=0, end=600, step=20)
	currFatSlider = Slider(title="Base fat (g)", value=inputDict['currFat'], start=0, end=120, step=5)
	intFatSlider = Slider(title="Int fat (g)", value=inputDict['intFat'], start=0, end=120, step=5)
	currProtnSlider = Slider(title="Base protein (g)", value=inputDict['currProtn'], start=0, end=120, step=5)
	intProtnSlider = Slider(title="Int protein (g)", value=inputDict['intProtn'], start=0, end=120, step=5)

	currMaleableInputs = [currWeightSlider, currWaterSlider, currAlcohol_freqeuncySelector, currDailyScreenTimeSlider,
	currActivityLevelSlider, currHours_of_sleepSlider, currCarboSlider, currFatSlider, currProtnSlider]

	# currMaleableInputsLayout = column(Div(text='Current lifestyle',style={'font-size': '200%', 'font-weight':'bold', 'color': 'black'}),
	# currWeightSlider, currSmokerCheckbox, currWaterSlider, currAlcohol_freqeuncySelector, currDailyScreenTimeSlider,
	# currActivityLevelSlider, currHours_of_sleepSlider, currCarboSlider, currFatSlider, currProtnSlider)

	currMaleableInputsLayout = column(Div(text='Current weight/lifestyle',style={'font-size': '200%', 'font-weight':'bold', 'color': 'black'}),
	currWeightSlider, currDailyScreenTimeSlider,currCarboSlider, currFatSlider, currProtnSlider)
	
	intMaleableInputs = [intWeightSlider, intSmokerCheckbox, intWaterSlider, intAlcohol_freqeuncySelector, intDailyScreenTimeSlider,
	intActivityLevelSlider, intHours_of_sleepSlider, intCarboSlider, intFatSlider, intProtnSlider]

	# intMaleableInputsLayout = column(Div(text='Intervention',style={'font-size': '200%', 'font-weight':'bold', 'color': 'black'}),
	# intWeightSlider, intSmokerCheckbox, intWaterSlider, intAlcohol_freqeuncySelector, intDailyScreenTimeSlider,
	# intActivityLevelSlider, intHours_of_sleepSlider, intCarboSlider, intFatSlider, intProtnSlider)

	intMaleableInputsLayout = column(Div(text='Intervention',style={'font-size': '200%', 'font-weight':'bold', 'color': 'red'}),
	intWeightSlider, intDailyScreenTimeSlider, intCarboSlider, intFatSlider, intProtnSlider)

	return currMaleableInputs, currMaleableInputsLayout, intMaleableInputs, intMaleableInputsLayout, currWeightSlider, currSmokerCheckbox, currWaterSlider, currAlcohol_freqeuncySelector, currDailyScreenTimeSlider, currActivityLevelSlider, currHours_of_sleepSlider, currCarboSlider, currFatSlider, currProtnSlider, intWeightSlider, intSmokerCheckbox, intWaterSlider, intAlcohol_freqeuncySelector, intDailyScreenTimeSlider, intActivityLevelSlider, intHours_of_sleepSlider, intCarboSlider, intFatSlider, intProtnSlider   

	# manipulableInputs = row(column(currBloodInputFields), column(row1, row2, row3, row4, row5, row6, row7, row8, row9, row10))
	# manipulableInputs = row(Div(text='',width =300), column(row0,row1, row2, row3, row4, row5, row6, row7, row8, row9, row10))
	# row(Div(text='Current:', style={'font-size': '200%', 'font-weight':'bold', 'color': 'black'}))

[fixedInputs_andBP_layout, modelSelector, ageSlider, sexSelector, heightSlider, locationSelector, medicalConditionSelector,
pregnantCheckbox, diabetesCheckbox, highBPCheckbox, heart_attackCheckbox, internal_bleedingCheckbox, 
systolText, diastolText] = createFixedInputs_andBP()

[currMaleableInputs, currMaleableInputsLayout, intMaleableInputs, intMaleableInputsLayout, currWeightSlider, currSmokerCheckbox, currWaterSlider, currAlcohol_freqeuncySelector, currDailyScreenTimeSlider,
currActivityLevelSlider, currHours_of_sleepSlider, currCarboSlider, currFatSlider, currProtnSlider, intWeightSlider, intSmokerCheckbox, intWaterSlider, intAlcohol_freqeuncySelector, intDailyScreenTimeSlider,
intActivityLevelSlider, intHours_of_sleepSlider, intCarboSlider, intFatSlider, intProtnSlider] = createMaleableInputs()

selectorsAndSliders = [modelSelector, ageSlider, sexSelector, locationSelector, medicalConditionSelector, heightSlider, 
intWeightSlider, intWaterSlider, intAlcohol_freqeuncySelector, intDailyScreenTimeSlider,
intActivityLevelSlider, intHours_of_sleepSlider, intCarboSlider, intFatSlider, intProtnSlider]


checkboxes = [pregnantCheckbox, diabetesCheckbox, highBPCheckbox, heart_attackCheckbox, internal_bleedingCheckbox]


########### Calculate values


def getInputs():
	modelName = modelSelector.value
	Age = ageSlider.value
	Sex = sexSelector.value
	Location = locationSelector.value
	Medical_condition = medicalConditionSelector.value
	Height = heightSlider.value
	currWeight = currWeightSlider.value
	intWeight = intWeightSlider.value
	currCups_water_daily = currWaterSlider.value
	intCups_water_daily = intWaterSlider.value
	currAlcohol_frequency = currAlcohol_freqeuncySelector.value
	intAlcohol_frequency = intAlcohol_freqeuncySelector.value
	currDailyScreenTime = currDailyScreenTimeSlider.value
	intDailyScreenTime = intDailyScreenTimeSlider.value
	currHours_of_sleep = currHours_of_sleepSlider.value
	intHours_of_sleep = intHours_of_sleepSlider.value
	currCarbo = currCarboSlider.value
	intCarbo = intCarboSlider.value
	currFat = currFatSlider.value
	intFat = intFatSlider.value
	currProtn = currProtnSlider.value
	intProtn = intProtnSlider.value
	currActivityLevel = currActivityLevelSlider.value
	intActivityLevel = intActivityLevelSlider.value

	if pregnantCheckbox.active:
		Pregnant = 1
	else: 
		Pregnant = 0

	if diabetesCheckbox.active:
		Diabetes = 1
	else: 
		Diabetes = 0

	if highBPCheckbox.active:
		High_BP = 1
	else: 
		High_BP = 0

	if heart_attackCheckbox.active:
		Heart_attack = 1
	else: 
		Heart_attack = 0

	if internal_bleedingCheckbox.active:
		Internal_bleeding = 1
	else: 
		Internal_bleeding = 0

	if currSmokerCheckbox.active:
		currSmoker = 1
	else: 
		currSmoker = 0

	if intSmokerCheckbox.active:
		intSmoker = 1
	else: 
		intSmoker = 0

	inputDict = dict(modelName = modelName, Age = Age, Sex = Sex,
										 Location = Location,
										 Medical_condition = Medical_condition, Pregnant = Pregnant, Diabetes = Diabetes,
										 High_BP = High_BP, Heart_attack = Heart_attack, Internal_bleeding = Internal_bleeding,
										 Height = Height, currWeight = currWeight, intWeight = intWeight,
										 currSmoker = currSmoker, intSmoker = intSmoker,
										 currCups_water_daily= currCups_water_daily, intCups_water_daily= intCups_water_daily,
										 currAlcohol_frequency=currAlcohol_frequency, intAlcohol_frequency=intAlcohol_frequency,
										 currDailyScreenTime = currDailyScreenTime, intDailyScreenTime = intDailyScreenTime,
										 currHours_of_sleep = currHours_of_sleep, intHours_of_sleep = intHours_of_sleep,
										 currCarbo = currCarbo, intCarbo = intCarbo,
										 currFat = currFat, intFat = intFat,
										 currProtn = currProtn, intProtn = intProtn,
										 currActivityLevel = currActivityLevel, intActivityLevel = intActivityLevel)
	# print(inputDict)
	return inputDict

def estBloodChanges(inputDict):

	[currentValues, futureValues] = hF.parseInputs(inputDict,inputFeatures)
	currExpectation = trainedModels[modelSelector.value].predict(currentValues.ravel().reshape(1, -1))
	# print(np.round(currExpectation,2))
	futurePrediction = trainedModels[modelSelector.value].predict(futureValues.ravel().reshape(1, -1))
	expectedRelChange = futurePrediction/currExpectation
	
	expectedRelChange_blood = expectedRelChange[0][0:-2]
	expectedRelChange_bloodPressure = expectedRelChange[0][-2:]
	est_blood_y = currExpectation[0][0:-2]
	est_bloodPressure_y = currExpectation[0][-2:]
	return expectedRelChange_blood, expectedRelChange_bloodPressure, est_blood_y, est_bloodPressure_y

def update_plots0():
	inputDict = getInputs()
	currBlood = np.round(np.array([insText.value, tgText.value, hdl_cText.value, ldl_cText.value, ureaText.value, uaText.value, apo_aText.value, lp_aText.value, hs_crpText.value, 
		creText.value, apo_bText.value, mgText.value, fetText.value, 
		hgbText.value, wbcText.value, rbcText.value, pltText.value, glu_fieldText.value,
		hba1cText.value, tpText.value, albText.value, glucoseText.value, tcText.value, altText.value, 
		trfText.value, trf_rText.value]).astype(float), decimals = 3)

	# expectedRelChange_blood, expectedRelChange_bloodPressure, currExpectation_blood, currExpectation_bloodPressure = estBloodChanges(inputDict) 
	# print(inputDict)
	# print(currExpectation_blood)
	#########
	# Estimation data: 
	expectedRelChange_blood, expectedRelChange_bloodPressure, est_blood_y, est_bloodPressure_y = estBloodChanges(inputDict) 
	estimatedBlood = [str(np.round(num, decimals = 2))+unitDictVals[i] for i,num in enumerate(est_blood_y)]

	est_blood_y = list(est_blood_y/medianBlood)
	est_blood_x = range(len(est_blood_y))
	est_bloodPressure_y = list(est_bloodPressure_y)
	est_bloodPressure_x = np.array(range(len(est_bloodPressure_y)))
	estBloodSource.data=dict(est_blood_x=est_blood_x, est_blood_y=est_blood_y, values = estimatedBlood)
	estBloodPressureSource.data = dict(est_bloodPressure_x=est_bloodPressure_x, est_bloodPressure_y=est_bloodPressure_y)

	# Prediction data:

	predict_bloodVariables = list(np.array(responseVariables[:-2]))
	predict_blood_y0 = currBlood/medianBlood#[~np.isnan(currBlood)]
	predict_blood_y1 = (expectedRelChange_blood.ravel()*currBlood)/medianBlood
	predictedLevel = expectedRelChange_blood.ravel()*currBlood
	predictedBlood = [str(np.round(num, decimals = 2))+unitDictVals[i] for i,num in enumerate(predictedLevel)]
	currentBlood = [str(np.round(num, decimals = 2))+unitDictVals[i] for i,num in enumerate(currBlood)]
	rel_change = [str(np.round(num, decimals = 2))+'%' for i,num in enumerate(100*(expectedRelChange_blood.ravel()-1))]

	predict_blood_x = np.array(range(len(predict_bloodVariables)))

	predict_bloodPressure_y = expectedRelChange_bloodPressure.ravel()*[float(systolText.value), float(diastolText.value)]
	predict_bloodPressure_x = np.array(range(len(predict_bloodPressure_y)))# y3 = expectedRelChange_weight.ravel()
	predict_bloodPressureSource.data =dict(predict_bloodPressure_x=predict_bloodPressure_x, 
		predict_bloodPressure_y0= np.array([float(systolText.value), float(diastolText.value)]),
		predict_bloodPressure_y1=predict_bloodPressure_y)

	predictBloodSource0.data=dict(predict_blood_x=predict_blood_x, 
	predict_blood_y0=predict_blood_y0,
	predict_blood_y1 = predict_blood_y1, values= currentBlood,
	rel_change = ['']*len(responseVariables[:-2]))

	predictBloodSource.data=dict(predict_blood_x=predict_blood_x, 
	predict_blood_y0=predict_blood_y0,
	predict_blood_y1 = predict_blood_y1, values= predictedBlood, rel_change = rel_change)

	# predictBloodSource.data=dict(predict_blood_x=predict_blood_x, 
	# predict_blood_y0=predict_blood_y0,
	# predict_blood_y1 = predict_blood_y1, predictedBlood = predictedBlood, rel_change = rel_change)

	# Plot 
	estPlot_blood.circle('est_blood_x', 'est_blood_y', source=estBloodSource, size=10, line_width=3, line_alpha=0.6, color = 'black') # Baseline
	estPlot_bloodPressure.circle('est_bloodPressure_x', 'est_bloodPressure_y', source=estBloodPressureSource, size = 10, line_width=3, line_alpha=0.6, color = 'black')

	# predictPlot_blood.add_glyph(predictBloodSource, predictGlyphs)
	predictPlot_blood.circle('predict_blood_x', 'predict_blood_y1', source=predictBloodSource, size=10, line_width=3, line_alpha=0.6, color = 'red') # Intervention
	predictPlot_blood.circle('predict_blood_x', 'predict_blood_y0', source=predictBloodSource0, size=10, line_width=3, line_alpha=0.6, color = 'black') # Baseline

	predictPlot_bloodPressure.circle('predict_bloodPressure_x', 'predict_bloodPressure_y1', source=predict_bloodPressureSource, size = 10, line_width=3, line_alpha=0.6, color = 'red')
	predictPlot_bloodPressure.circle('predict_bloodPressure_x', 'predict_bloodPressure_y0', source=predict_bloodPressureSource, size = 10, line_width=3, line_alpha=0.6, color = 'black')
	return

def update_plots():
	global in_progress

	if in_progress:
		pass
	else: 
		in_progress = True
		print("Adding callback")
		doc.add_next_tick_callback(tick_callback_function)

def tick_callback_function():
	global in_progress
	update_plots0()
	# predictPlot_blood.xaxis.major_label_overrides = dict(zip(predict_blood_x,predict_bloodVariables))
	#########
	in_progress = False
	return 


# inputDict = getInputs()
####### Plotting section: 
in_progress = True
# Glyph data:
estGlyph_x = list(range(len(responseVariables[:-2])))
estGlyph_y = np.array((medianBlood)/medianBlood)
estGlyph_w = np.array(np.ones(26)*.75)
estGlyph_h = np.array((blood_std/4)/medianBlood)
estGlyphSource = ColumnDataSource(data=dict(estGlyph_x=estGlyph_x, 
	estGlyph_y=estGlyph_y, 
	estGlyph_w=estGlyph_w, 
	estGlyph_h=estGlyph_h, 
	values = responseVariables[:-1]))

predictGlyph_x = list(range(len(responseVariables[:-2])))
predictGlyph_y = np.array((medianBlood)/medianBlood)
predictGlyph_w = np.array(np.ones(26)*.75)
predictGlyph_h = np.array((blood_std/4)/medianBlood)

predictGlyphSource = ColumnDataSource(data = dict(
	predictGlyph_x=predictGlyph_x, 
	predictGlyph_y=predictGlyph_y, 
	predictGlyph_w=predictGlyph_w, 
	predictGlyph_h=predictGlyph_h,
	values = responseVariables[:-2],
	rel_change = ['']*len(responseVariables[:-2])))

################# Format Plots:
estimatedTOOLTIPS=[
    ("", "@values")
]
predictedTOOLTIPS=[
    ("", "@values"),
    ("", "@rel_change")
]
estPlot_blood = figure(plot_height=500, plot_width=800, title="Estimated blood test",
							tools="pan,reset,save",
							x_range=[-1, 27], y_range=[.25, 2.25],
							toolbar_location=None, tooltips=estimatedTOOLTIPS)
estPlot_blood.title.text_font_size = '14pt'
estPlot_blood.xaxis.ticker = FixedTicker(ticks=list(estGlyph_x))
estPlot_blood.xaxis.axis_label = 'Biomarker'
estPlot_blood.xaxis.axis_label_text_font_size = '16pt'
estPlot_blood.xaxis.major_label_overrides = dict(zip(estGlyph_x,responseVariables[0:-2]))
estPlot_blood.xaxis.major_label_text_font_size = '12pt'
estPlot_blood.xaxis.major_label_orientation = 3.14/4
estPlot_blood.yaxis.axis_label = 'Level relative to median'
estPlot_blood.yaxis.axis_label_text_font_size = '12pt'

estPlot_bloodPressure = figure(plot_height=500, plot_width=100, title="BP",
							tools="pan,reset,save",
							x_range=[-1, 2], y_range=[60, 150],
							toolbar_location=None)
estPlot_bloodPressure.title.text_font_size = '14pt'
estPlot_bloodPressure.xaxis.ticker = list(range(2))
estPlot_bloodPressure.xaxis.major_label_overrides = dict(zip(range(2),responseVariables[-2:]))
estPlot_bloodPressure.xaxis.major_label_orientation = 3.14/2
estPlot_bloodPressure.xaxis.axis_label_text_font_size = '16pt'
estPlot_bloodPressure.xaxis.major_label_text_font_size = '12pt'
estPlot_bloodPressure.xgrid.visible = False
estPlot_bloodPressure.yaxis.axis_label = 'Pressure (mmHg)'
estPlot_bloodPressure.yaxis.axis_label_text_font_size = '12pt'

#

predictPlot_blood = figure(plot_height=500, plot_width=800, title="Relative change in blood test",
							tools="pan,reset,save", y_range=[.25, 2.25],
							toolbar_location=None, tooltips = predictedTOOLTIPS)
predictPlot_blood.title.text_font_size = '14pt'
predictPlot_blood.xaxis.ticker = predictGlyph_x
predictPlot_blood.xaxis.axis_label = 'Biomarker'
predictPlot_blood.xaxis.axis_label_text_font_size = '16pt'
predictPlot_blood.yaxis.axis_label = 'Level relative to median'
predictPlot_blood.yaxis.axis_label_text_font_size = '12pt'
predictPlot_blood.xaxis.major_label_overrides = dict(zip(predictGlyph_x,responseVariables[:-2]))
predictPlot_blood.xaxis.major_label_text_font_size = '12pt'
predictPlot_blood.xaxis.major_label_orientation = 3.14/4

predictPlot_bloodPressure = figure(plot_height=500, plot_width=100, title="BP",
							tools="pan,reset,save",
							x_range=[-1, 2], y_range=[60, 150],
							toolbar_location=None)
predictPlot_bloodPressure.title.text_font_size = '14pt'
predictPlot_bloodPressure.xaxis.ticker = list(range(2))
predictPlot_bloodPressure.xaxis.major_label_overrides = dict(zip(range(2),responseVariables[-2:]))
predictPlot_bloodPressure.xaxis.major_label_orientation = 3.14/2
predictPlot_bloodPressure.xaxis.axis_label_text_font_size = '16pt'
predictPlot_bloodPressure.xaxis.major_label_text_font_size = '12pt'
predictPlot_bloodPressure.yaxis.axis_label = 'Pressure (mmHg)'
predictPlot_bloodPressure.yaxis.axis_label_text_font_size = '12pt'
predictPlot_bloodPressure.xgrid.visible = False

estBloodPressureSource = ColumnDataSource(data=dict(est_bloodPressure_x=[], est_bloodPressure_y=[]))

predict_bloodPressureSource = ColumnDataSource(data=dict(predict_bloodPressure_x=[], 
	predict_bloodPressure_y0= [],
	predict_bloodPressure_y1=[], values = [], rel_change = []))


predictBloodSource0 = ColumnDataSource(data=dict(predict_blood_x=[], 
	predict_blood_y0=[],
	predict_blood_y1 = [], values= [], rel_change = []))

predictBloodSource = ColumnDataSource(data=dict(predict_blood_x=[], 
	predict_blood_y0=[],
	predict_blood_y1 = [], values= [], rel_change = []))

####### Plot Data: 

estBloodSource = ColumnDataSource(data=dict(est_blood_x=[],
	est_blood_y=[], 
	values = []))
estPlot_blood.circle('est_blood_x', 'est_blood_y', source=estBloodSource, size=10, line_width=3, line_alpha=0.6, color = 'black') # Baseline
estGlyphs = Rect(x="estGlyph_x", y="estGlyph_y", width="estGlyph_w", height="estGlyph_h", fill_color="grey", fill_alpha = .2)
estPlot_blood.add_glyph(estGlyphSource, estGlyphs)
estPlot_bloodPressure.circle('est_bloodPressure_x', 'est_bloodPressure_y', source=estBloodPressureSource, size = 10, line_width=3, line_alpha=0.6, color = 'black')


predictGlyphs = Rect(x="predictGlyph_x", y="predictGlyph_y", width="predictGlyph_w", height="predictGlyph_h", fill_color="grey", fill_alpha = .2)
predictPlot_blood.add_glyph(predictGlyphSource, predictGlyphs)
predictPlot_blood.circle('predict_blood_x', 'predict_blood_y1', source=predictBloodSource, size=10, line_width=3, line_alpha=0.6, color = 'red') # Intervention
predictPlot_blood.circle('predict_blood_x', 'predict_blood_y0', source=predictBloodSource0, size=10, line_width=3, line_alpha=0.6, color = 'black') # Baseline

predictPlot_bloodPressure.circle('predict_bloodPressure_x', 'predict_bloodPressure_y1', source=predict_bloodPressureSource, size = 10, line_width=3, line_alpha=0.6, color = 'red')
predictPlot_bloodPressure.circle('predict_bloodPressure_x', 'predict_bloodPressure_y0', source=predict_bloodPressureSource, size = 10, line_width=3, line_alpha=0.6, color = 'black')

in_progress = False
update_plots0()


############# Update functionality

def update_currWeightSlider(attrname, old, new):
	intWeightSlider.value = currWeightSlider.value
	update_plots()

def update_currWaterSlider(attrname, old, new):
	intWaterSlider.value = currWaterSlider.value
	update_plots()

def update_currAlcohol_freqeuncySelector(attrname, old, new):
	intAlcohol_freqeuncySelector.value = currAlcohol_freqeuncySelector.value
	update_plots()

def update_currDailyScreenTimeSlider(attrname, old, new):
	intDailyScreenTimeSlider.value = currDailyScreenTimeSlider.value 
	update_plots()

def update_currActivityLevelSlider(attrname, old, new):
	intActivityLevelSlider.value = currActivityLevelSlider.value 
	update_plots()

def update_currHours_of_sleepSlider(attrname, old, new):
	intHours_of_sleepSlider.value = currHours_of_sleepSlider.value 
	update_plots()

def update_currCarboSlider(attrname, old, new):
	intCarboSlider.value = currCarboSlider.value 
	update_plots()

def update_currFatSlider(attrname, old, new):
	intFatSlider.value = currFatSlider.value 
	update_plots()

def update_currProtnSlider(attrname, old, new):
	intProtnSlider.value = currProtnSlider.value
	update_plots()

def update_data(attrname, old, new):
	update_plots()

def update_blood(attrname, old, new):
	inputDict = getInputs()
	expectedRelChange_blood, expectedRelChange_bloodPressure, currExpectation_blood, currExpectation_bloodPressure = estBloodChanges(inputDict) 
	for i, w in zip(range(len(currBloodInputFields)), currBloodInputFields): 
		w.title = w.title.split()[0] + ' est: ' +  str(np.round(currExpectation_blood[i], decimals = 2))
	update_plots()

def update_data2(attrname):
	update_plots()

def clearBloodFields(attrname):
	for w in currBloodInputFields:
		w.value = '0'

clearButton.on_click(clearBloodFields)

# currWeightSlider.on_change('value', update_currWeightSlider)
# currWaterSlider.on_change('value', update_currWaterSlider)
# currAlcohol_freqeuncySelector.on_change('value', update_currAlcohol_freqeuncySelector)
# currDailyScreenTimeSlider.on_change('value', update_currDailyScreenTimeSlider)
# currActivityLevelSlider.on_change('value', update_currActivityLevelSlider)
# currHours_of_sleepSlider.on_change('value', update_currHours_of_sleepSlider)
# currCarboSlider.on_change('value', update_currCarboSlider)
# currFatSlider.on_change('value', update_currFatSlider)
# currProtnSlider.on_change('value', update_currProtnSlider)


selectorsAndSliders = [modelSelector, ageSlider, sexSelector, locationSelector, medicalConditionSelector, heightSlider, 
currWeightSlider, currWaterSlider, currAlcohol_freqeuncySelector, currDailyScreenTimeSlider,
currActivityLevelSlider, currHours_of_sleepSlider, currCarboSlider, currFatSlider, currProtnSlider,
intWeightSlider, intWaterSlider, intAlcohol_freqeuncySelector, intDailyScreenTimeSlider,
intActivityLevelSlider, intHours_of_sleepSlider, intCarboSlider, intFatSlider, intProtnSlider]


for w in selectorsAndSliders:
	w.on_change('value', update_data)

for w in currBloodInputFields:
	w.on_change('value', update_blood)

for w in checkboxes:
	w.on_click(update_data2)



tab1 = Panel(child = column(row(estPlot_blood,estPlot_bloodPressure), row(Div(), Div())), title = 'Estimation')
tab2 = Panel(child=column(currBloodInputLayout, row(systolText, diastolText)), title="My blood test")
tab3 = Panel(child = column(row(predictPlot_blood, predictPlot_bloodPressure), row(Div(text = ' ', width = 150), intMaleableInputsLayout)), title = 'My blood predictions')

tabs = Tabs(tabs=[tab1, tab2, tab3])

layout = column(row(column(fixedInputs_andBP_layout,Div(height = 130), currMaleableInputsLayout),tabs, width=200, height = 600))

doc = curdoc()
doc.title = "Health Forecaster"
doc.add_root(layout) 

# output_file("Test.html", title="Testing")
# show(layout)
