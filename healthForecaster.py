import numpy as np
from sas7bdat import SAS7BDAT
import glob
import pandas as pd
from sklearn import preprocessing
from sas7bdat import SAS7BDAT
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import utils, model_selection, metrics, linear_model, neighbors, ensemble


def convertAllCHSData(year = [], onlySubjectsWBiomarkers = 0):
    if onlySubjectsWBiomarkers:
        print('Only obtaining data for subjects/households with biomarker data.')
    dataDirs = glob.glob('./data/Master*')
    for dir in dataDirs:
        SASfiles = glob.glob(dir + '/*.sas7bdat')
        for SASfile in SASfiles: 
            convertSASfile(SASfile, year)

def convertSASfile(inputFullPath, year = [], onlySubjectsWBiomarkers = 0):
    print('Converting ' + inputFullPath)
    df = SAS2DataFrame(inputFullPath, year = year)
    outputName = inputFullPath.split('/')[-1].split('.')[0] 
    outputDir = '/'.join(inputFullPath.split('/')[0:-2])
    if year:
        outputFullPath = outputDir + '/' + outputName
        outputFullPath = outputFullPath + '_' + str(year) + 'only' + '.csv'   
    else:
        outputFullPath = outputDir + '/' + outputName + '.csv'
    
    if onlySubjectsWBiomarkers: 
        subjectsWithBiomarkers = pd.read_csv('./data/subjectsWithBiomarkers.csv')
        tmp = set(df.columns)
        identifyingFields = list(tmp.intersection(set(subjectsWithBiomarkers.columns)))
        if not identifyingFields:
            print('No identifying fields found.')
            return
        elif identifyingFields.count('idind'):
            selFactor = 'idind'
            selidinds = list(set(df[selFactor]).intersection(set(subjectsWithBiomarkers[selFactor])))
            selIdxs = [a in selidinds for a in df[selFactor]]
            df = df[selIdxs]
        elif identifyingFields.count('hhid'):
            selFactor = 'hhid'
            selidinds = list(set(df[selFactor]).intersection(set(subjectsWithBiomarkers[selFactor])))
            selIdxs = [a in selidinds for a in df[selFactor]]
            df = df[selIdxs]
        elif identifyingFields.count('commid'):
            selFactor = 'commid'
            selidinds = list(set(df[selFactor]).intersection(set(subjectsWithBiomarkers[selFactor])))
            selIdxs = [a in selidinds for a in df[selFactor]]
            df = df[selIdxs]
    
    print(str(df.shape[0]) + ' valid rows')
    df.to_csv(outputFullPath)
    return
        
def SAS2DataFrame(inputFullPath, year = []):
    with SAS7BDAT(inputFullPath, skip_header=False) as reader:
        df = reader.to_data_frame()
        df.columns = [col.lower() for col in df.columns]
        if (not not year) & any(df.columns == 'wave'):
            df = df[df['wave'] == year]
    return df

def getSurveyData():
    ''' Gets relevant survey data for dHealth project
    i.e. survey data for subjects that have biomarker data
    '''
    surveyPath = './data/Master_ID_201908/surveys_pub_12.sas7bdat'
    surveyData = SAS2DataFrame(surveyPath)
    surveyData = surveyData[(surveyData['biomaker'] == 1) & (surveyData['wave'] == 2009)]
    return surveyData

def getBiomarkerData():
    surveyData = getSurveyData()
    biomarkerPath = './data/Master_Biomarker_2009/biomarker_09.sas7bdat'
    biomarkerData = SAS2DataFrame(biomarkerPath)
    ids1 = set(biomarkerData.idind)
    ids2 = set(surveyData.idind)
    excludeIds = list(ids1.difference(ids2))
    for id in excludeIds: 
        tmp = list(biomarkerData.idind)
        idx = tmp.index(id)
        biomarkerData = biomarkerData.drop(idx)
    return biomarkerData

def createSubjectsWithBiomarkersCSV():
    surveyData = getSurveyData()
    surveyData = surveyData.iloc[:,[0,1,5,3]]
    surveyData.columns = ['idind', 'hhid', 'commid', 'Age']
    surveyData.to_csv('./data/subjectsWithBiomarkers.csv')

# createSubjectsWithBiomarkersCSV()
featureMap = pd.read_csv('featureTableMap.csv')
subjects = pd.read_csv('./data/subjectsWithBiomarkers.csv',usecols = ['idind','Age']) # Could add others too'hhid','commid'

def createGenderCSV():
    print('Extracting gender data...')
    subjects = pd.read_csv('./data/subjectsWithBiomarkers.csv',usecols = ['idind','hhid','commid'])
    subjects = subjects.astype({'idind': 'int',
                                'hhid': 'int',
                                'commid': 'int'})
    def getGender(subjectIdx, idind_1, idind_2, sex_1, sex_2):
        gender = np.nan
        if subjects.idind[subjectIdx] in idind_1:
            idx = idind_1.index(subjects.idind[subjectIdx])
            gender = int(sex_1[idx])
        elif subjects.idind[subjectIdx] in idind_2: 
            idx = idind_2.index(subjects.idind[subjectIdx])
            gender = int(sex_2[idx])
        else:     
            gender = np.nan

        if gender == 1:
            gender = int(1)
        elif gender == 2:
            gender = 0
        if subjectIdx % 500 == 0: 
            print(str(100*subjectIdx/9548) + '% complete')       
        return gender
    
    relations = pd.read_csv('./data/relationmast_pub_00_2009only.csv')
    idind_1 = list(relations.idind_1)
    idind_2 = list(relations.idind_2)
    sex_1 = list(relations.sex_1)
    sex_2 = list(relations.sex_2)
    
    gender = [getGender(i, idind_1, idind_2, sex_1, sex_2) for i in range(len(subjects))]
    d = {'idind': subjects.idind, 'Sex': gender}
    df = pd.DataFrame(data=d)
    df.to_csv('./data/gender.csv')

def createSleep_ScreenTimeCSV():
    sleep_screenTime = pd.read_csv('./data/pact_12_2009only.csv',usecols = ['idind', 'u324', 'u339','u340_mn', 'u341_mn','u508', 'u509_mn','u510_mn','u345','u346_mn', 'u347_mn'])
    sleep_screenTime.columns = ['idind', 'Hours_of_sleep', 'watchTV','TVhours_week','TVhours_weekend','goesOnline','online_week','online_weekend', 'play_videoGames', 'videoGames_week', 'videoGames_weekend']
    sleep_screenTime = sleep_screenTime.replace({'watchTV':{9:1,np.nan:1}, 'goesOnline':{9:0,np.nan:0}, 'play_videoGames':{9:0,np.nan:0}, 'Hours_of_sleep':{-9: np.nan}})
    sleep_screenTime = sleep_screenTime.fillna(sleep_screenTime.median())
    sleep_screenTime_subjects= list(sleep_screenTime.idind)

    
    def getDailyScreenTime(subjectIdx):
        weeklyScreenTime = 0        
        if subjects.idind[subjectIdx] in sleep_screenTime_subjects: 
            idx = sleep_screenTime_subjects.index(subjects.idind[subjectIdx])
        else:
            return np.nan
        
        if sleep_screenTime.watchTV[idx]:
            weeklyScreenTime = weeklyScreenTime + sleep_screenTime.TVhours_week[idx] + sleep_screenTime.TVhours_weekend[idx]
        else:
            pass
        
        if sleep_screenTime.goesOnline[idx]:
            weeklyScreenTime = weeklyScreenTime + sleep_screenTime.online_week[idx] + sleep_screenTime.online_weekend[idx]
        else:
            pass
        
        if sleep_screenTime.play_videoGames[idx]:
            weeklyScreenTime = weeklyScreenTime + sleep_screenTime.videoGames_week[idx] + sleep_screenTime.videoGames_weekend[idx]
        else:
            pass
        return np.round(weeklyScreenTime/7)
    
    def getDailySleepTime(subjectIdx):      
        if subjects.idind[subjectIdx] in sleep_screenTime_subjects: 
            idx = sleep_screenTime_subjects.index(subjects.idind[subjectIdx])
        else:
            return np.nan
        return sleep_screenTime.Hours_of_sleep[idx]
    
    subjects = pd.read_csv('./data/subjectsWithBiomarkers.csv',usecols = ['idind'])
    Daily_screen_time = [getDailyScreenTime(i) for i in range(len(subjects))]
    Hours_of_sleep = [getDailySleepTime(i) for i in range(len(subjects))]
    d = {'idind': subjects.idind, 'Daily_screen_time': Daily_screen_time, 'Hours_of_sleep': Hours_of_sleep}
    df = pd.DataFrame(data=d)
    df.to_csv('./data/sleep_screentime.csv')
    return df    

# Define these variables for default inputs for the functions below:

def preprocessRawChinaHealthStudyData():
    createSubjectsWithBiomarkersCSV()
    convertAllCHSData(year = 2009, onlySubjectsWBiomarkers = 1)
    createGenderCSV()
    createSleep_ScreenTimeCSV()


def getAndMergeTables(subjects = subjects, tableNum = 1):
    newDF = pd.read_csv('./data/'+featureMap['tablename'][tableNum],usecols = eval(featureMap['varnames'][tableNum]))
    
    newDF.columns = eval(featureMap['newnames'][tableNum])

    try: 
        replaceDict = eval(featureMap['replacements'][tableNum])
        print('This should not work for surveys')
        newDF.replace(replaceDict, inplace = True)
    except: 
        print('Could not replace values or none exists.')

    subjects = pd.merge(subjects,newDF,how='left', on ='idind')
    print(list(newDF.columns))
    print(subjects.columns)
    return subjects

def createDataTable():
    subjects = pd.read_csv('./data/subjectsWithBiomarkers.csv',usecols = ['idind','Age'])
    print('Adding demographic info')  
    for i in range(1,4):
        print('Adding ' + featureMap['tablename'][i])
        subjects = getAndMergeTables(subjects = subjects, tableNum = i)
    
    print('One-hot-encoding medical conditions...')   
    # One-hot-encode medical conditions: 
    medicalConditions = subjects['Medical_condition'].fillna('noReport')
    medicalConditions = medicalConditions.fillna('noReport')
    medicalConditions = pd.DataFrame(medicalConditions)
    enc = preprocessing.OneHotEncoder(categories = "auto")
    enc.fit(medicalConditions)
    data = enc.transform(medicalConditions).toarray()
    columnNames = enc.categories_[0]
    medicalConditions = pd.DataFrame(data,columns=columnNames)
    # Replace old medical condition column to one-hot-encoded vars:
    subjects.drop('Medical_condition', axis=1, inplace=True)
    subjects=pd.concat([subjects,medicalConditions], axis=1, ignore_index=False)

    # Add physical exam: 
    print('Adding lifestyle features...')
    i = 4
    print('Adding ' + featureMap['tablename'][i])
    subjects = getAndMergeTables(subjects = subjects, tableNum = i)


    # Add lifestyle features: 
    print('Adding lifestyle features...')
    for i in range(5,featureMap.shape[0]-1):
        print('Adding ' + featureMap['tablename'][i])
        subjects = getAndMergeTables(subjects = subjects, tableNum = i)
        
    
    print('Adding reponse variables...') 
    # Add the response variables (biomarker levels):
    i = featureMap.shape[0]-1
    print('Adding ' + featureMap['tablename'][i])
    subjects = getAndMergeTables(subjects = subjects, tableNum = i)

    # Median impute missing data: 
    subjects = subjects.fillna(subjects.median())
    #Change data types: 
    subjects = subjects.astype({'idind': 'int',
                    'Sex': 'int',
                    'Urban': 'int',
                    'Activity_level': 'int'}) 

    return subjects

def shuffleAndSplit(featureMatrix, targetMatrix, test_size=.2, n_splits=5):
    # Shuffle datasets:
    X,Y = utils.shuffle(featureMatrix,targetMatrix, random_state = 0) 

    # Split X and y into training and test sets (80% Train : 20% Test):
    X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(
        X, Y, random_state = 0, test_size = test_size)

    cv=model_selection.KFold(n_splits = n_splits, shuffle = False)
    return X_Train, X_Test, Y_Train, Y_Test, cv

def showDataSplits(Y_Train, Y_Test, cv):
    ''' Helper function to show how the data was split
    '''
    fig, ax = plt.subplots(figsize = (12,3))
    plt.xlim(0,len(Y_Train)+len(Y_Test))
    plt.ylim(0,cv.n_splits+1.5)
    ax.set_title('Training and Validation splits \n (after shuffling)')
    plt.xlabel('Dataset indicies')
    yticklabels= []; 
    offset = -.4
    i = 0
    for train_idxs, cval_idxs in cv.split(Y_Train):
        # training data: 
        i += 1
        start = (min(train_idxs),i+offset)
        width = max(train_idxs)-min(train_idxs)
        if i == 1:
            ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'c', label = 'CV_train'))
        ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'c'))
        # cross-validation data: 
        start = (min(cval_idxs),i+offset)
        width = max(cval_idxs)-min(cval_idxs)
        if i == 1:
            ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'orange', label = 'CV_validation')) 
        ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'orange'))
        yticklabels.append('Cross validation_' + str(i))
    
    start = (0,cv.n_splits+1+offset)
    width = len(Y_Train)
    ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'g', label = 'Train')) 
    start = (len(Y_Train),cv.n_splits+1+offset)
    width = len(Y_Train)
    ax.add_patch(mpl.patches.Rectangle(start, width = width, height = .8, color = 'r', label = 'Test')) 
    yticklabels.append('Final test')
    
    #Format plot
    plt.yticks(np.arange(1,cv.n_splits+2),yticklabels)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()

def createHealthForecasterModels():
    import pickle
    ## Aggregate relevant data for ML:
    data = createDataTable()
    fixedFactors = ['Age', 'Sex', 'Urban', 'ENT', 'OBGYN', 'Old_age_midLife_syndrome', 'alcohol_poisoning',
                'dermatological', 'digestive', 'endocrine', 'heart', 'hematological', 'infectious_parasitic', 'injury',
                'muscular_rheumatological', 'neurological', 'noDiagnosis', 'noReport', 'other', 'pyschiatric', 'respiratory',
                'sexualDysfunction', 'tumor', 'unknown', 'urinary', 'High_BP', 'Diabetes', 'Heart_attack', 'Internal_bleeding', 
                'Pregnant','Height']
    fixedFactorIdxs = [list(data.columns).index(varName) for varName in fixedFactors]

    lifestyleFactors = ['Smoker', 'Cups_water_daily', 'Alcohol_frequency', 'Weight', 'Kcal', 'Carbs', 'Fat', 'Protein', 'Activity_level', 'Daily_screen_time', 'Hours_of_sleep']
    lifestyleFactorIdxs = [list(data.columns).index(varName) for varName in lifestyleFactors]
    responseVariables = ['Insulin','Triglycerides','HDL_C', 'LDL_C','Urea', 'Uric_acid', 'APO_A', 'Lipoprotein_A','High_sensitivity_CRP', 'Creatinine',
                         'APO_B', 'Mg', 'Ferritin', 'Hemoglobin', 'White_blood_cell',
                         'Red_blood_cell', 'Platelet', 'Glucose_field','HbA1c', 'Total_protein','Albumin', 'Glucose',
                         'Total_cholestorol', 'Alanine_AT', 'Transferrin', 'Transferrin_receptor','Systol', 'Diastol']

    responseVariableIdxs = [list(data.columns).index(varName) for varName in responseVariables]

    fatRelatedIdxs = [responseVariables.index('APO_A'), 
                      responseVariables.index('Lipoprotein_A'), 
                      responseVariables.index('HDL_C'),
                      responseVariables.index('LDL_C'),
                      responseVariables.index('APO_B'),
                      responseVariables.index('Triglycerides'),
                      responseVariables.index('Total_cholestorol')]
    gluRelatedIdxs = [responseVariables.index('Insulin'),
                          responseVariables.index('HbA1c'),
                          responseVariables.index('Glucose')]

    inputFeatures = fixedFactors + lifestyleFactors
    X = data[inputFeatures].to_numpy()
    Y = data[responseVariables].to_numpy()

    # Y_zscore = (Y-np.mean(Y,axis=0))/np.std(Y,axis=0)

    # X_Train, X_Test, Y_Train, Y_Test, cv = shuffleAndSplit(X, Y, test_size=.2, n_splits=5)
    # X_Train, X_Test, Y_Train_zscore, Y_Test_zscore, cv = shuffleAndSplit(X, Y_zscore, test_size=.2, n_splits=5)

    ## Create a second model to predict weight:

    # fixedFactors2 = ['age', 'sex', 'urban', 'ENT', 'OBGYN', 'Old_age_midLife_syndrome', 'alcohol_poisoning',
    #                 'dermatological', 'digestive', 'endocrine', 'heart', 'hematological', 'infectious_parasitic', 'injury',
    #                 'muscular_rheumatological', 'neurological', 'noDiagnosis', 'noReport', 'other', 'pyschiatric', 'respiratory',
    #                 'sexualDysfunction', 'tumor', 'unknown', 'urinary', 'highBP', 'diabetes', 'heart_attack', 'internal_bleeding', 
    #                 'pregnant','height']
    # fixedFactorIdxs2 = [list(data.columns).index(varName) for varName in fixedFactors]

    # lifestyleFactors2 = ['smoker', 'cups_water_daily', 'Alcohol_frequency', 'kcal', 'carbo', 'fat', 'protn', 'Activity_level', 'Daily_screen_time', 'Hours_of_sleep']
    # lifestyleFactorIdxs2 = [list(data.columns).index(varName) for varName in lifestyleFactors]
    # responseVariables2 = ['weight']
    # responseVariableIdxs2 = [list(data.columns).index(varName) for varName in responseVariables2]
    
    # inputFeatures2 = fixedFactors2+lifestyleFactors2

    # X2 = data[fixedFactors2 + lifestyleFactors2].to_numpy()
    # Y2 = data[responseVariables2].to_numpy()

    # X_Train2, X_Test2, Y_Train2, Y_Test2, cv = shuffleAndSplit(X2, Y2, test_size=.2, n_splits=5)

    models = dict(ols=linear_model.LinearRegression(),
              lasso=linear_model.Lasso(alpha=0.75),
              ridge=linear_model.Ridge(alpha=0.75),
              elastic=linear_model.ElasticNet(alpha=0.1, l1_ratio=0.75),
              randomForest = ensemble.RandomForestRegressor(random_state=0, 
                                                           max_features = 'auto', 
                                                           min_samples_leaf = 50, #max_depth = 3,
                                                           n_estimators = 200)
             )

    # Also define models to predict z_score Target Matrix
    # models_zscore = dict(ols=linear_model.LinearRegression(),
    #              lasso=linear_model.Lasso(alpha=.5),
    #              ridge=linear_model.Ridge(alpha=.5),
    #              elastic=linear_model.ElasticNet(alpha=.5, l1_ratio=0.5),
    #              randomForest = ensemble.RandomForestRegressor(random_state=0, 
    #                                                           max_features = 'auto', 
    #                                                           min_samples_leaf = 10,
    #                                                           n_estimators = 200)

    # weightModel = dict(ols=linear_model.LinearRegression(),
    #             lasso=linear_model.Lasso(alpha=.5),
    #             ridge=linear_model.Ridge(alpha=.5),
    #             elastic=linear_model.ElasticNet(alpha=.5, l1_ratio=0.5),
    #             randomForest = ensemble.RandomForestRegressor(random_state=0, 
    #                                                            max_features = 'auto', 
    #                                                            min_samples_leaf = 10,
    #                                                            n_estimators = 200))
    # print('Training trainedWeightBPModels')
    # trainedWeightModels = {}
    # for name, mdl in weightModel.items(): 
    #     print('Training ' + str(name) + '...')
    #     trainedWeightModels.update({name : mdl.fit(X2,Y2.ravel())})
    # print('finished')


    # Train models
    print('Training trainedModels')
    trainedModels = {}
    for name, mdl in models.items(): 
        print('Training ' + str(name) + '...')
        trainedModels.update({name : mdl.fit(X,Y)})
    print('finished')
    # pickle.dump([trainedModels, trainedWeightModels, inputFeatures, responseVariables, inputFeatures2, responseVariables2], open("models.p", "wb"))
    pickle.dump([trainedModels, inputFeatures, responseVariables, data], open("models.p", "wb"))

    # return trainedModels, trainedWeightModels, inputFeatures, responseVariables, inputFeatures2, responseVariables2
    return trainedModels, inputFeatures, responseVariables

def parseInputs(inputDict,inputFeatures):
    # inputValues = np.zeros(len(inputFeatures)) 
    currentValues = np.zeros(len(inputFeatures)) 
    futureValues = np.zeros(len(inputFeatures)) 
    # Age
    currentValues[inputFeatures.index('Age')] = inputDict['Age']
    futureValues[inputFeatures.index('Age')] = inputDict['Age']
    # Sex
    if inputDict['Sex'] == 'M':
        currentValues[inputFeatures.index('Sex')] = 1
        futureValues[inputFeatures.index('Sex')] = 1
    else: 
        currentValues[inputFeatures.index('Sex')] = 0
        futureValues[inputFeatures.index('Sex')] = 0

    # Location:
    if inputDict['Location'] == 'Urban':
        currentValues[inputFeatures.index('Urban')] = 1
        futureValues[inputFeatures.index('Urban')] = 1
    else: 
        currentValues[inputFeatures.index('Urban')] = 0
        futureValues[inputFeatures.index('Urban')] = 0

    # Physical exam/Medical Conditions: 
    
    currentValues[inputFeatures.index('Height')] = inputDict['Height']*2.54
    futureValues[inputFeatures.index('Height')] = inputDict['Height']*2.54
    
    currentValues[inputFeatures.index(inputDict['Medical_condition'])] = 1
    futureValues[inputFeatures.index(inputDict['Medical_condition'])] = 1

    if inputDict['Pregnant']: 
        currentValues[inputFeatures.index('Pregnant')] = 1
        futureValues[inputFeatures.index('Pregnant')] = 1
    if inputDict['Diabetes']: 
        currentValues[inputFeatures.index('Diabetes')] = 1
        futureValues[inputFeatures.index('Diabetes')] = 1
    if inputDict['High_BP']:
        currentValues[inputFeatures.index('High_BP')] = 1
        futureValues[inputFeatures.index('High_BP')] = 1
    if inputDict['Heart_attack']:
        currentValues[inputFeatures.index('Heart_attack')] = 1
        futureValues[inputFeatures.index('Heart_attack')] = 1
    if inputDict['Internal_bleeding']:
        currentValues[inputFeatures.index('Internal_bleeding')] = 1
        futureValues[inputFeatures.index('Internal_bleeding')] = 1
    
    # currentValues = futureValues = inputValues # This may have done some weird cloning thing?

    ### Current lifestyle
    
    # Habits:
    if inputDict['currAlcohol_frequency'] == 'daily':
        currentValues[inputFeatures.index('Alcohol_frequency')] = 1
    elif inputDict['currAlcohol_frequency'] == '3-4 times a week':
        currentValues[inputFeatures.index('Alcohol_frequency')] = 2
    elif inputDict['currAlcohol_frequency'] == 'Once or twice a week':
        currentValues[inputFeatures.index('Alcohol_frequency')] = 3
    elif inputDict['currAlcohol_frequency'] == 'Once or twice a month':
        currentValues[inputFeatures.index('Alcohol_frequency')] = 4
    elif inputDict['currAlcohol_frequency'] == 'No more than once a month':
        currentValues[inputFeatures.index('Alcohol_frequency')] = 5
    else: 
        currentValues[inputFeatures.index('Alcohol_frequency')] = 3

    currentValues[inputFeatures.index('Cups_water_daily')] = inputDict['currCups_water_daily']
    
    if inputDict['currSmoker']: 
        currentValues[inputFeatures.index('Smoker')] = 1
    
    # Diet/Weight:
    currentValues[inputFeatures.index('Kcal')] = inputDict['currCarbo']*4 + inputDict['currProtn']*4 + inputDict['currFat']*9 #currKcal
    currentValues[inputFeatures.index('Carbs')] = inputDict['currCarbo']
    currentValues[inputFeatures.index('Fat')] = inputDict['currFat']
    currentValues[inputFeatures.index('Protein')] = inputDict['currProtn']
    
    # Activity
    currentValues[inputFeatures.index('Activity_level')] = inputDict['currActivityLevel']
    currentValues[inputFeatures.index('Daily_screen_time')] = inputDict['currDailyScreenTime']
    currentValues[inputFeatures.index('Hours_of_sleep')] = inputDict['currHours_of_sleep']
    
    if 'Weight' in inputFeatures:
        currentValues[inputFeatures.index('Weight')] = inputDict['currWeight']/2.205
        
    ### Lifestyle intervention
    
    # Habits:
    if inputDict['intAlcohol_frequency'] == 'daily':
        futureValues[inputFeatures.index('Alcohol_frequency')] = 1
    elif inputDict['intAlcohol_frequency'] == '3-4 times a week':
        futureValues[inputFeatures.index('Alcohol_frequency')] = 2
    elif inputDict['intAlcohol_frequency'] == 'Once or twice a week':
        futureValues[inputFeatures.index('Alcohol_frequency')] = 3
    elif inputDict['intAlcohol_frequency'] == 'Once or twice a month':
        futureValues[inputFeatures.index('Alcohol_frequency')] = 4
    elif inputDict['intAlcohol_frequency'] == 'No more than once a month':
        futureValues[inputFeatures.index('Alcohol_frequency')] = 5
    else: 
        futureValues[inputFeatures.index('Alcohol_frequency')] = 3

    futureValues[inputFeatures.index('Cups_water_daily')] = inputDict['intCups_water_daily']
    
    if inputDict['intSmoker']: 
        futureValues[inputFeatures.index('Smoker')] = 1
    
    # Diet/Weight:
    futureValues[inputFeatures.index('Kcal')] = inputDict['intCarbo']*4 + inputDict['intProtn']*4 + inputDict['intFat']*9 #currKcal
    futureValues[inputFeatures.index('Carbs')] = inputDict['intCarbo']
    futureValues[inputFeatures.index('Fat')] = inputDict['intFat']
    futureValues[inputFeatures.index('Protein')] = inputDict['intProtn']
    
    # Activity
    futureValues[inputFeatures.index('Activity_level')] = inputDict['intActivityLevel']
    futureValues[inputFeatures.index('Daily_screen_time')] = inputDict['intDailyScreenTime']
    futureValues[inputFeatures.index('Hours_of_sleep')] = inputDict['intHours_of_sleep']
    
    if 'Weight' in inputFeatures:
        futureValues[inputFeatures.index('Weight')] = inputDict['intWeight']/2.205
    
    return currentValues, futureValues

def plotSubjectModelPrediction(trainedModels, X, Y, responseVariables, modelName = 'randomForest', subjectIdx = 3):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(11, 5))
    y_predict = trainedModels[modelName].predict(X[subjectIdx,:].reshape(1, -1))
    plt.scatter(range(0,26), Y[subjectIdx,:].T,color = 'b',label = 'actual')
    plt.scatter(range(0,26), y_predict.T,color = 'r',label = 'prediction')
    plt.xticks(range(0,26))
    plt.xticks(rotation='vertical')
    ax.set_xticklabels(responseVariables)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()


## Helper functions

def update_progress(numerator, denominator=1, taskName = 'Progress'):
    from IPython.display import clear_output
    bar_length = 20
    if isinstance(numerator, int):
        numerator = float(numerator)
    if not isinstance(numerator, float):
        numerator = 0
    if numerator/denominator < 0:
        numerator = 0
    if numerator/denominator >= 1:
        numerator = denominator
    block = int(round(bar_length * (numerator/denominator)))
    clear_output(wait = True)
    text = taskName + ": [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), (numerator/denominator) * 100)
    print(text)

