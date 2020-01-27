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
    surveyData.iloc[:,[0,1,5,3]].to_csv('./data/subjectsWithBiomarkers.csv')
    
def createGenderCSV():
    print('Extracting gender data...')
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
    
    subjects = pd.read_csv('./data/subjectsWithBiomarkers.csv',usecols = ['idind','hhid','commid'])
    subjects = subjects.astype({'idind': 'int',
                                'hhid': 'int',
                                'commid': 'int'})
    relations = pd.read_csv('./data/relationmast_pub_00_2009only.csv')
    idind_1 = list(relations.idind_1)
    idind_2 = list(relations.idind_2)
    sex_1 = list(relations.sex_1)
    sex_2 = list(relations.sex_2)
    
    gender = [getGender(i, idind_1, idind_2, sex_1, sex_2) for i in range(len(subjects))]
    d = {'idind': subjects.idind, 'sex': gender}
    df = pd.DataFrame(data=d)
    df.to_csv('./data/gender.csv')

# Define these variables for default inputs for the functions below:
featureMap = pd.read_csv('featureTableMap.csv')
subjects = pd.read_csv('./data/subjectsWithBiomarkers.csv',usecols = ['idind','age']) # Could add others too'hhid','commid'

def preprocessRawChinaHealthStudyData():
    createSubjectsWithBiomarkersCSV()
    convertAllCHSData(year = 2009, onlySubjectsWBiomarkers = 1)
    createGenderCSV()


def getAndMergeTables(subjects = subjects, tableNum = 2):
    newDF = pd.read_csv('./data/'+featureMap['tablename'][tableNum],usecols = eval(featureMap['varnames'][tableNum]))
    newDF.columns = eval(featureMap['newnames'][tableNum])
    try: 
        replaceDict = eval(featureMap['replacements'][tableNum])
        newDF = newDF.replace(replaceDict)
    except: 
        pass
    subjects = pd.merge(subjects,newDF,how='left', on ='idind')
    return subjects

def createDataTable():
    subjects = pd.read_csv('./data/subjectsWithBiomarkers.csv',usecols = ['idind','age'])
    print('Adding demographic info')  
    for i in range(1,4):
        print('Adding ' + featureMap['tablename'][i])
        subjects = getAndMergeTables(subjects = subjects, tableNum = i)
    
    print('One-hot-encoding medical conditions...')   
    # One-hot-encode medical conditions: 
    medicalConditions = subjects['medicalCondition'].fillna('noReport')
    medicalConditions = medicalConditions.fillna('noReport')
    medicalConditions = pd.DataFrame(medicalConditions)
    enc = preprocessing.OneHotEncoder(categories = "auto")
    enc.fit(medicalConditions)
    data = enc.transform(medicalConditions).toarray()
    columnNames = enc.categories_[0]
    medicalConditions = pd.DataFrame(data,columns=columnNames)
    # Replace old medical condition column to one-hot-encoded vars:
    subjects.drop('medicalCondition', axis=1, inplace=True)
    subjects=pd.concat([subjects,medicalConditions], axis=1, ignore_index=False)

    # Add lifestyle features: 
    print('Adding lifestyle features...')
    for i in range(4,featureMap.shape[1]):
        print('Adding ' + featureMap['tablename'][i])
        subjects = getAndMergeTables(subjects = subjects, tableNum = i)
        


    
    print('Adding reponse variables...') 
    # Add the response variables (biomarker levels):
    i = featureMap.shape[1]
    print('Adding ' + featureMap['tablename'][i])
    subjects = getAndMergeTables(subjects = subjects, tableNum = i)

    # Median impute missing data: 
    subjects = subjects.fillna(subjects.median())

    #Change data types: 
    subjects = subjects.astype({'idind': 'int',
                    'sex': 'int',
                    'urban': 'int',
                    'activityLevel': 'int'}) 

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

def plotSubjectModelPrediction(trainedModels, X, Y, responseVariables, modelName = 'ridge', subjectIdx = 3):
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



