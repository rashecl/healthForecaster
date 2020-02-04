import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Select, Dropdown, CheckboxGroup, Div, FixedTicker, Panel, Tabs
from bokeh.plotting import figure
from bokeh.models.glyphs import Rect

import matplotlib
matplotlib.use('TkAgg')
import healthForecaster as hF
import pickle 
# pickle.dump([trainedModels, inputFeatures, responseVariables], open("models.p", "wb"))
[trainedModels, inputFeatures, responseVariables] = pickle.load( open( "models.p", "rb" ) )
# Alternatively extracts, parses, and trains the models:
# [trainedModels, inputFeatures, responseVariables] = hF.createHealthForecasterModels()
inputDict = dict(modelName = 'randomForest', Age = 65, Sex = 'F',
                       Location = 'urban',
                       Medical_condition = 'noReport', Pregnant = 0, Diabetes = 0, High_BP = 0, Heart_attack = 0, Internal_bleeding = 0,
                       Height = 157.48, currWeight = 72.56, 
                       currSmoker = 0, intSmoker = 0,
                       currCups_water_daily= 3,
                       currAlcohol_frequency=2, 
                       currDailyScreenTime = 240, 
                       currHours_of_sleep = 8, 
                       currCarbo = 500, 
                       currFat = 100, 
                       currProtn = 100, 
                       currActivityLevel = 1) #currWeight = 64.2, intWeight = 64.2,
inputDict.update(intWeight = currWeight, intCups_water_daily= currCups_water_daily, intAlcohol_frequency=currAlcohol_frequency,
  intDailyScreenTime = currDailyScreenTime, intHours_of_sleep = currHours_of_sleep,
  intCarbo = currCarbo, intFat = currFat, intProtn = currProtn, intActivityLevel = currActivityLevel)


medianBlood = np.array([5.20, 296.00, 1.08, 78.00, 1.00, 83.00, 1.38,  2.83,  0.86,  0.94, 73.72, 10.52,140.00,  6.10,  4.65,215.00,  5.02,  5.50, 77.10, 47.40,  5.08,  1.22,  4.68, 18.00,284.00,  1.36])
initialBlood = np.array([5.58, 274.79, 1.19, 499.83, 2.54, 82.29, 1.54, 3.24, 0.99, 0.93,137.6,16.56,130.72, 5.95, 4.43, 207.97, 5.2, 5.65, 77.64, 46.83 , 5.27, 1.93, 5.2, 20.15, 271.23, 1.39])

ureaText       =    TextInput(title="urea     ", value=str(initialBlood[0]    ))
uaText         =    TextInput(title="ua       ", value=str(initialBlood[1]    ))
apo_aText      =    TextInput(title="apo_a    ", value=str(initialBlood[2]    ))
lp_aText       =    TextInput(title="lp_a     ", value=str(initialBlood[3]    ))
hs_crpText     =    TextInput(title="hs_crp   ", value=str(initialBlood[4]    ))
creText        =    TextInput(title="cre      ", value=str(initialBlood[5]    ))
hdl_cText      =    TextInput(title="hdl_c    ", value=str(initialBlood[6]    ))
ldl_cText      =    TextInput(title="ldl_c    ", value=str(initialBlood[7]    ))
apo_bText      =    TextInput(title="apo_b    ", value=str(initialBlood[8]    ))
mgText         =    TextInput(title="mg       ", value=str(initialBlood[9]    ))
fetText        =    TextInput(title="fet      ", value=str(initialBlood[10]   ))
insText        =    TextInput(title="ins      ", value=str(initialBlood[11]   ))
hgbText        =    TextInput(title="hgb      ", value=str(initialBlood[12]   ))
wbcText        =    TextInput(title="wbc      ", value=str(initialBlood[13]   ))
rbcText        =    TextInput(title="rbc      ", value=str(initialBlood[14]   ))
pltText        =    TextInput(title="plt      ", value=str(initialBlood[15]   ))
glu_fieldText  =    TextInput(title="glu_field", value=str(initialBlood[16]   ))
hba1cText      =    TextInput(title="hba1c    ", value=str(initialBlood[17]   ))
tpText         =    TextInput(title="tp       ", value=str(initialBlood[18]   ))
albText        =    TextInput(title="alb      ", value=str(initialBlood[19]   ))
glucoseText    =    TextInput(title="glucose  ", value=str(initialBlood[20]   ))
tgText         =    TextInput(title="tg       ", value=str(initialBlood[21]   ))
tcText         =    TextInput(title="tc       ", value=str(initialBlood[22]   ))
altText        =    TextInput(title="alt      ", value=str(initialBlood[23]   ))
trfText        =    TextInput(title="trf      ", value=str(initialBlood[24]   ))
trf_rText      =    TextInput(title="trf_r    ", value=str(initialBlood[25]   ))


def estRelativeChange(inputDict):

  [currentValues, futureValues] = hF.parseInputs(inputDict,inputFeatures)
  currExpectation = trainedModels['randomForest'].predict(currentValues.ravel().reshape(1, -1))
  print(np.round(currExpectation,2))
  futurePrediction = trainedModels['randomForest'].predict(futureValues.ravel().reshape(1, -1))
  expectedRelChange = futurePrediction/currExpectation
  
  expectedRelChange_blood = expectedRelChange[0][0:-2]
  expectedRelChange_bloodPressure = expectedRelChange[0][-2:]
  # [currentValues2, futureValues2] = hF.parseInputs(inputDict,inputFeatures2)
  # currExpectation_Weight = trainedWeightModels['randomForest'].predict(currentValues2.ravel().reshape(1, -1))*2.205
  # futurePrediction_Weight = trainedWeightModels['randomForest'].predict(futureValues2.ravel().reshape(1, -1))
  
  # expectedRelChange_Weight = futurePrediction_WeightBP/currExpectation_WeightBP-1
  # futurePrediction_Weight[0][-1] = futurePrediction_WeightBP[0][-1]*2.205

  return expectedRelChange_blood, expectedRelChange_bloodPressure

expectedRelChange_blood, expectedRelChange_bloodPressure = estRelativeChange(inputDict) 

x0 = np.array(range(0,len(responseVariables[0:-2])))
y0 = initialBlood/medianBlood
x1 = np.array(range(0,len(responseVariables[0:-2])))
x2 = np.array(range(2))    
# x3 = np.array(range(1)) 
y1 = (expectedRelChange_blood.ravel()*initialBlood)/medianBlood
y2 = expectedRelChange_bloodPressure.ravel()
# y3 = expectedRelChange_weight.ravel()

source0 = ColumnDataSource(data=dict(x0=x0, y0=y0))
source1 = ColumnDataSource(data=dict(x1=x1, y1=y1))
source2 = ColumnDataSource(data=dict(x2=x2, y2=y2))
# source3 = ColumnDataSource(data=dict(x3=x3, y3=y3))

###### Bloot test scatter
plot1 = figure(plot_height=500, plot_width=800, title="Relative change in blood test",
              tools="pan,reset,save",
              x_range=[-1, 27], y_range=[.25, 2.25],
              toolbar_location=None)
plot1.title.text_font_size = '14pt'
# , x_range=[0, 4*np.pi], y_range=[-2.5, 2.5]
plot1.circle('x1', 'y1', source=source1, size=10, line_width=3, line_alpha=0.6, color = 'red') # Intervention
plot1.circle('x0', 'y0', source=source0, size=10, line_width=3, line_alpha=0.6, color = 'black') # Baseline

x_glyph = np.arange(0,26,1)
y_glyph = np.array([  5.2 , 296.  ,   1.08,  78.  ,   1.  ,  83.  ,   1.38,   2.83,
         0.86,   0.94,  73.72,  10.52, 140.  ,   6.1 ,   4.65, 215.  ,
         5.02,   5.5 ,  77.1 ,  47.4 ,   5.08,   1.22,   4.68,  18.  ,
       284.  ,   1.36])/medianBlood
w = np.array(np.ones(26)*.75)
h = np.array([1.6000e+00, 1.0415e+02, 3.7000e-01, 2.2328e+02, 8.7400e+00,
       2.2810e+01, 5.0000e-01, 9.9000e-01, 2.7000e-01, 1.0000e-01,
       1.7972e+02, 2.3310e+01, 2.0300e+01, 1.9600e+00, 6.8000e-01,
       7.0160e+01, 1.4100e+00, 8.9000e-01, 5.1600e+00, 3.4700e+00,
       1.4300e+00, 1.4500e+00, 1.0200e+00, 1.9190e+01, 5.5650e+01,
       7.1000e-01])/(medianBlood*4)

glyphSource_chol = ColumnDataSource(data=dict(x4=x_glyph, y4=y_glyph, w=w, h=h))
glyph = Rect(x="x4", y="y4", width="w", height="h", fill_color="grey", fill_alpha = .2)
plot1.add_glyph(glyphSource_chol, glyph)

plot1.xaxis.ticker = FixedTicker(ticks=list(range(0,len(responseVariables[0:-2]))))
plot1.xaxis.axis_label = 'Biomarker'
plot1.xaxis.axis_label_text_font_size = '16pt'
plot1.xaxis.major_label_overrides = dict(zip(range(0,len(responseVariables[0:-2])),responseVariables[0:-2]))
plot1.xaxis.major_label_text_font_size = '12pt'
plot1.xaxis.major_label_orientation = 3.14/4
tab1 = Panel(child=plot1, title="Blood test")

plot2 = figure(plot_height=500, plot_width=100, title="BP",
              tools="pan,reset,save",
              x_range=[-1, 2], y_range=[.75, 1.25],
              toolbar_location=None)
plot2.title.text_font_size = '14pt'
plot2.circle('x2', 'y2', source=source2, size = 10, line_width=3, line_alpha=0.6, color = 'red')
plot2.xaxis.ticker = list(range(0,2))
plot2.xaxis.major_label_overrides = dict(zip(range(0,2),responseVariables[-2:]))
plot2.xaxis.major_label_orientation = 3.14/4
plot2.xaxis.axis_label_text_font_size = '16pt'
plot2.xaxis.major_label_text_font_size = '12pt'
plot2.xgrid.visible = False

# plot3 = figure(plot_height=500, plot_width=100, title="Weight",
#               tools="pan,reset,save",
#               x_range=[-.75, .75], y_range=[-.25, .25],
#               toolbar_location=None)
# plot3.scatter('x3', 'y3', source=source3, line_width=3, line_alpha=0.6)
# plot3.xaxis.ticker = list(range(0,2))
# plot3.xaxis.major_label_overrides = {0 : responseVariables[-1]}
# plot3.xaxis.major_label_orientation = 3.14/4
# plot3.xgrid.visible = False
# plot3.xaxis.axis_label_text_font_size = '16pt'
# plot3.xaxis.major_label_text_font_size = '12pt'


############# Widgets

modelList = [("Random forest", "randomForest"), ("Linear model", "ols"), ("Ridge regression", "ridge"), ("Elastic Net", "elastic")]
modelSelector = Select(title="Model", options=modelList, value = 'randomForest')
ageSlider = Slider(title="Age", value=inputDict['age'], start=0, end=100, step=1)
sexSelector = Select(title="Sex", options=['M', 'F'], value = inputDict['sex'])
heightSlider=Slider(title="Height (inches)", value=inputDict['height']/2.54, start=0, end=90, step=1)
locationSelector = Select(title="Location", options=['urban', 'rural'], value = 'urban')
# medicalConditionList = [('noReport','noReport'), ('ENT', 'ENT'), ('OBGYN','OBGYN'),
#        ('Old_age_midLife_syndrome','Old_age_midLife_syndrome'), ('alcohol_poisoning', 'alcohol_poisoning'), ('dermatological','dermatological'),
#        ('digestive','digestive'), ('endocrine','endocrine'), ('heart','heart'), ('hematological','hematological'),
#        ('infectious_parasitic','infectious_parasitic'), ('injury','injury'), ('muscular_rheumatological','muscular_rheumatological'),
#        ('neurological','neurological'), ('noDiagnosis','noDiagnosis'), ('other','other'), ('pyschiatric','pyschiatric'),
#        ('respiratory','respiratory'), ('sexualDysfunction','sexualDysfunction'), ('tumor','tumor'), ('unknown','unknown'), ('urinary','urinary')]
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

fixedInputs = column(modelSelector, Div(text='Background:',width=300, style={'font-size': '200%', 'font-weight':'bold', 'color': 'black'}), 
  ageSlider, sexSelector, heightSlider, locationSelector, medicalConditionSelector,
  pregnantCheckbox, diabetesCheckbox, highBPCheckbox, heart_attackCheckbox, internal_bleedingCheckbox)

# Maleable inputs:
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

currCarboSlider = Slider(title="Base carbs (g)", value=inputDict['currCarbo'], start=1, end=600, step=20)
intCarboSlider = Slider(title="Int carbs (g)", value=inputDict['intCarbo'], start=1, end=600, step=20)
currFatSlider = Slider(title="Base fat (g)", value=inputDict['currFat'], start=1, end=120, step=5)
intFatSlider = Slider(title="Int fat (g)", value=inputDict['intFat'], start=1, end=120, step=5)
currProtnSlider = Slider(title="Base protein (g)", value=inputDict['currProtn'], start=1, end=150, step=5)
intProtnSlider = Slider(title="Int protein (g)", value=inputDict['intProtn'], start=1, end=150, step=5)

row1 = row(currWeightSlider, Div(text='',width=100),intWeightSlider)
row2 = row(currSmokerCheckbox, Div(text='',width=100), intSmokerCheckbox)
row3 = row(currWaterSlider, Div(text='',width=100), intWaterSlider)
row4 = row(currAlcohol_freqeuncySelector, Div(text='',width=100), intAlcohol_freqeuncySelector)
row5 = row(currDailyScreenTimeSlider, Div(text='',width=100), intDailyScreenTimeSlider)
row6 = row(currActivityLevelSlider, Div(text='',width=100), intActivityLevelSlider) 
row7 = row(currHours_of_sleepSlider, Div(text='',width=100), intHours_of_sleepSlider)
row8 = row(currCarboSlider, Div(text='',width=100), intCarboSlider)
row9 = row(currFatSlider, Div(text='',width=100), intFatSlider)
row10 = row(currProtnSlider, Div(text='',width=100), intProtnSlider)

currBloodInputFields = [ureaText, uaText, apo_aText, lp_aText, hs_crpText, creText, hdl_cText,
 ldl_cText, apo_bText, mgText, fetText, insText, hgbText, wbcText, rbcText, pltText, glu_fieldText,
 hba1cText, tpText, albText, glucoseText, tgText, tcText, altText, trfText, trf_rText]


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
tmp8 = row(currBloodInputFields[r*3], currBloodInputFields[r*3+1], Div())
currBloodInputLayout = column(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8)


tab2 = Panel(child=currBloodInputLayout, title="My blood test")
tabs = Tabs(tabs=[tab1, tab2])

# manipulableInputs = row(column(currBloodInputFields), column(row1, row2, row3, row4, row5, row6, row7, row8, row9, row10))
manipulableInputs = row(Div(text='',width=300), column(row1, row2, row3, row4, row5, row6, row7, row8, row9, row10))

selectorsAndSliders = [modelSelector, ageSlider, sexSelector, locationSelector, medicalConditionSelector, heightSlider,
currWeightSlider, intWeightSlider, currWaterSlider , intWaterSlider, currAlcohol_freqeuncySelector,
intAlcohol_freqeuncySelector, currDailyScreenTimeSlider, intDailyScreenTimeSlider,
currHours_of_sleepSlider, intHours_of_sleepSlider, currCarboSlider, intCarboSlider,  
currFatSlider, intFatSlider, currProtnSlider, intProtnSlider, currActivityLevelSlider, intActivityLevelSlider]

checkboxes = [pregnantCheckbox, diabetesCheckbox, highBPCheckbox, heart_attackCheckbox, internal_bleedingCheckbox]

def update_data(attrname, old, new):
  modelName = modelSelector.value
  age = ageSlider.value
  sex = sexSelector.value
  location = locationSelector.value
  medicalCondition = medicalConditionSelector.value
  height = heightSlider.value
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
    pregnant = 1
  else: 
    pregnant = 0

  if diabetesCheckbox.active:
    diabetes = 1
  else: 
    diabetes = 0

  if highBPCheckbox.active:
    highBP = 1
  else: 
    highBP = 0

  if heart_attackCheckbox.active:
    heart_attack = 1
  else: 
    heart_attack = 0

  if internal_bleedingCheckbox.active:
    internal_bleeding = 1
  else: 
    internal_bleeding = 0

  if currSmokerCheckbox.active:
    currSmoker = 1
  else: 
    currSmoker = 0

  if intSmokerCheckbox.active:
    intSmoker = 1
  else: 
    intSmoker = 0

    inputDict = dict(modelName = modelName, age = age, sex = sex,
                       location = location,
                       medicalCondition = medicalCondition, pregnant = pregnant, diabetes = diabetes,
                       highBP = highBP, heart_attack = heart_attack, internal_bleeding = internal_bleeding,
                       height = height, currWeight = currWeight, intWeight = intWeight,
                       currSmoker = currSmoker, intSmoker = intSmoker,
                       currCups_water_daily= currCups_water_daily, intCups_water_daily= intCups_water_daily,
                       currAlcohol_frequency=currAlcohol_frequency, intAlcohol_frequency=intAlcohol_frequency,
                       currDailyScreenTime = currDailyScreenTime, intDailyScreenTime = intDailyScreenTime,
                       currHours_of_sleep = currHours_of_sleep, intHours_of_sleep = intHours_of_sleep,
                       currCarbo = currCarbo, intCarbo = intCarbo,
                       currFat = currFat, intFat = intFat,
                       currProtn = currProtn, intProtn = intProtn,
                       currActivityLevel = currActivityLevel, intActivityLevel = intActivityLevel)

    expectedRelChange_blood, expectedRelChange_bloodPressure = estRelativeChange(inputDict) 
    x1 = np.array(range(0,len(responseVariables[0:-2])))
    x2 = np.array(range(2))    
    # x3 = np.array(range(1)) 
    currBlood = np.array([ureaText.value, uaText.value, apo_aText.value, lp_aText.value, hs_crpText.value, 
      creText.value, hdl_cText.value, ldl_cText.value, apo_bText.value, mgText.value, fetText.value, 
      insText.value, hgbText.value, wbcText.value, rbcText.value, pltText.value, glu_fieldText.value,
      hba1cText.value, tpText.value, albText.value, glucoseText.value, tgText.value, tcText.value, altText.value, 
      trfText.value, trf_rText.value]).astype(float)    
    x0 = np.array(range(0,len(responseVariables[0:-2])))
    y0 = currBlood / medianBlood 
    y1 = expectedRelChange_blood.ravel()*currBlood/medianBlood
    y2 = expectedRelChange_bloodPressure.ravel()
    source0.data = dict(x0=x0, y0=y0)    
    source1.data = dict(x1=x1, y1=y1)
    source2.data = dict(x2=x2, y2=y2)
    return 

def update_data2(attrname):
  modelName = modelSelector.value
  age = ageSlider.value
  sex = sexSelector.value
  location = locationSelector.value
  medicalCondition = medicalConditionSelector.value
  height = heightSlider.value
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
    pregnant = 1
  else: 
    pregnant = 0

  if diabetesCheckbox.active:
    diabetes = 1
  else: 
    diabetes = 0

  if highBPCheckbox.active:
    highBP = 1
  else: 
    highBP = 0

  if heart_attackCheckbox.active:
    heart_attack = 1
  else: 
    heart_attack = 0

  if internal_bleedingCheckbox.active:
    internal_bleeding = 1
  else: 
    internal_bleeding = 0

  if currSmokerCheckbox.active:
    currSmoker = 1
  else: 
    currSmoker = 0

  if intSmokerCheckbox.active:
    intSmoker = 1
  else: 
    intSmoker = 0


    inputDict = dict(modelName = modelName, age = age, sex = sex,
                       location = location,
                       medicalCondition = medicalCondition, pregnant = pregnant, diabetes = diabetes,
                       highBP = highBP, heart_attack = heart_attack, internal_bleeding = internal_bleeding,
                       height = height, currWeight = currWeight, intWeight = intWeight,
                       currSmoker = currSmoker, intSmoker = intSmoker,
                       currCups_water_daily= currCups_water_daily, intCups_water_daily= intCups_water_daily,
                       currAlcohol_frequency=currAlcohol_frequency, intAlcohol_frequency=intAlcohol_frequency,
                       currDailyScreenTime = currDailyScreenTime, intDailyScreenTime = intDailyScreenTime,
                       currHours_of_sleep = currHours_of_sleep, intHours_of_sleep = intHours_of_sleep,
                       currCarbo = currCarbo, intCarbo = intCarbo,
                       currFat = currFat, intFat = intFat,
                       currProtn = currProtn, intProtn = intProtn,
                       currActivityLevel = currActivityLevel, intActivityLevel = intActivityLevel)

    expectedRelChange_blood, expectedRelChange_bloodPressure = estRelativeChange(inputDict) 
    
    x1 = np.array(range(0,len(responseVariables[0:-2])))
    x2 = np.array(range(2))    
    # x3 = np.array(range(1))    
    currBlood = np.array([ureaText.value, uaText.value, apo_aText.value, lp_aText.value, hs_crpText.value, 
      creText.value, hdl_cText.value, ldl_cText.value, apo_bText.value, mgText.value, fetText.value, 
      insText.value, hgbText.value, wbcText.value, rbcText.value, pltText.value, glu_fieldText.value,
      hba1cText.value, tpText.value, albText.value, glucoseText.value, tgText.value, tcText.value, altText.value, 
      trfText.value, trf_rText.value]).astype(float)   
    x0 = np.array(range(0,len(responseVariables[0:-2])))
    y0 = currBlood / medianBlood 
    y1 = expectedRelChange_blood.ravel()*currBlood/medianBlood
    y2 = expectedRelChange_bloodPressure.ravel()
    source0.data = dict(x0=x0, y0=y0)    
    source1.data = dict(x1=x1, y1=y1)
    source2.data = dict(x2=x2, y2=y2)

for w in selectorsAndSliders + currBloodInputFields:
    w.on_change('value', update_data)

for w in checkboxes:
    w.on_click(update_data2)


curdoc().add_root(column(row(fixedInputs,tabs,plot2, width=200, height = 600), manipulableInputs)) # plot3 could be added
curdoc().title = "Health Forecaster"
# p = hplot(plot1, plot2)
# show(p)