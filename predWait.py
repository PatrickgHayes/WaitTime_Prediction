# IMPORTS
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
from sklearn import linear_model
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Read in data initial

# Function for reading in the json files
def parseJSON(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

data = parseJSON("merged_file.json")

# Take all the data from each queue and put it in one list
allData = []
for queue in data:
    for line in queue:
        allData.append(line)


# Preprocess data

# Convert the average wait time for each day to milliseconds adn convert the created at feature to a datetime
for line in allData:
    line['avg_wait_for_day'] = int(line['avg_wait_for_day'] * 60 * 1000)
    if type(line['created_at']) is int:
        line['created_at'] = datetime.datetime.fromtimestamp(int(line['created_at'])/1000).strftime('%Y-%m-%d')


# Get the average time it took to resolve a ticket per day for a queue
resolution_time = defaultdict(list)
resolution_tickets = defaultdict(int)

for line in allData:
    if not line['time_to_resolve'] is None:
        resolution_time[line['created_at'] + line['course']].append((line['daily_ticket_count'],line['time_to_resolve'])) 
        resolution_tickets[line['created_at'] + line['course']] += 1

running_daily_avg = dict()
for day in resolution_time:
    resolution_time[day].sort()
    recentTickets = [720000] * 5
    
    # Keep a queue of the last 10 tickets get rid of outliers
    for ticket in resolution_time[day]:
        if ticket[1] / (60*1000) >= 5 and ticket[1] / (60*1000) <= 45:
            recentTickets.append(ticket[1])
        if len(recentTickets) > 10:
            recentTickets.pop(0)
        running_daily_avg[day+str(ticket[0])] = sum(recentTickets) * 1.0 / len(recentTickets)

avg_resolution_time = dict()
for day in resolution_time:
    avg_resolution_time[day] = sum([b for a, b in resolution_time[day]]) / resolution_tickets[day]
    
tutorStats = defaultdict(int)
ticketCount = defaultdict(int)

for line in allData:
    if not line['time_to_resolve'] is None:
        tutorStats[line['tutor_id']] += (line['time_to_resolve'] * 1.0
                                        / avg_resolution_time[line['created_at'] + line['course']])
        ticketCount[line['tutor_id']] += 1
        
for tutor in tutorStats:
    tutorStats[tutor] = (tutorStats[tutor] * 1.0 / ticketCount[tutor])
    


# In[10]:

def ValidNeighbor(day, ticket_count):
    for i in range(1,6):
        if (day + str(ticket_count + i)) in running_daily_avg:
            return i
        elif (day + str(ticket_count - i)) in running_daily_avg:
            return -i
    return 0


# In[11]:

for line in allData:
    if (line['created_at'] + line['course'] + str(line['daily_ticket_count'])) in running_daily_avg:
        line['avg_daily_res_time'] = running_daily_avg[line['created_at'] + line['course']
                                                       + str(line['daily_ticket_count'])] 
        
    elif (ValidNeighbor(line['created_at'] + line['course'], line['daily_ticket_count']) != 0):
        line['avg_daily_res_time'] = running_daily_avg[line['created_at'] + line['course']
                                        + str(line['daily_ticket_count'] + ValidNeighbor(line['created_at']
                                        + line['course'], line['daily_ticket_count']))]  
        
    elif (line['created_at'] + line['course']) in avg_resolution_time:
        line['avg_daily_res_time'] = avg_resolution_time[line['created_at'] + line['course']]
        
    else:
        line['avg_daily_res_time'] = 12
        


# In[12]:

## Test that preprocessing worked
#for i in range(0,900):
#    print str(int(allData[i]['avg_daily_res_time'] / (60 * 1000))) + '   ---    ' + str(allData[i]['daily_ticket_count'])


# In[ ]:




# In[13]:

# Read in data after preprocessing


# In[14]:

Data100 = [line for line in allData if '100' in line['course']]
print len(Data100)


# In[15]:

Data12 = [line for line in allData if '12' in line['course']]
print len(Data12)


# In[ ]:




# In[ ]:

# Graphs to check for coorelation betweeen feature and error


# In[ ]:

#itr_list = [list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list()]
#for j in range(0,100):
#    random.shuffle(allData)
#    halfData = Data12[:len(allData)/5]
#
#    totalPredictedTime = [0] * 24
#    totalWaitTime = [0] * 24
#    numberOfTickets = [0] * 24
#
#    for line in halfData:
#        if len(line["tutors_on_duty"]) == 0:
#            totalPredictedTime[int(line['time_of_day'][:2])] += 6.0 * (line["position"] * 1.0) + 9.3
#        else:
#            totalPredictedTime[int(line['time_of_day'][:2])] += 6.0 * (line["position"] * 1.0 / len(line["tutors_on_duty"])) + 9.3
#        totalWaitTime[int(line['time_of_day'][:2])] += line['diff'] * 1.0 / (60 * 1000)
#        numberOfTickets[int(line['time_of_day'][:2])] += 1
#    
#    for i in range(0,24):
#        if numberOfTickets[i] == 0:
#            itr_list[i].append(0)
#        else:
#            itr_list[i].append((totalWaitTime[i] * 1.0 / numberOfTickets[i])
#                           - (totalPredictedTime[i] * 1.0 / numberOfTickets[i]))
#
#avgErr = [0] * 24
#stdErr = [0] * 24        
#for l in range(0,24):
#    stdErr[l] = np.std(np.array(itr_list[l]))
#    avgErr[l] = np.mean(np.array(itr_list[l]))
#dayOfWeek = range(0,24)
#print avgErr
#print stdErr
#
#plt.errorbar(dayOfWeek, avgErr, stdErr, linestyle='None', marker='^')
#plt.show()
#
#
#
#
## In[ ]:
#
#itr_list = [list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(), list() 
#           ]
#for j in range(0,10):
#    random.shuffle(allData)
#    halfData = allData[:len(allData)/5]
#
#    totalPredictedTime = [0] * 90
#    totalWaitTime = [0] * 90
#    numberOfTickets = [0] * 90
#
#    for line in halfData:
#        if line['position'] >= 90:
#            continue
#        if len(line["tutors_on_duty"]) == 0:
#            totalPredictedTime[0] += (4.07378565 * (len(line["tutors_on_duty"]) * 1.0) + 1.62954177 * (line["position"] * 1.0)
#                                        + 0.74638183)
#        else:
#            totalPredictedTime[len(line['tutors_on_duty'])] += (4.07378565 * (line["position"] * 1.0 / len(line["tutors_on_duty"])) 
#                                                            + 1.62954177 * (line["position"] * 1.0) + 0.74638183)
#        totalWaitTime[len(line['tutors_on_duty'])] += line['diff'] * 1.0 / (60 * 1000)
#        numberOfTickets[len(line['tutors_on_duty'])] += 1
#    
#    for i in range(0,90):
#        if numberOfTickets[i] == 0:
#            itr_list[i].append(0)
#        else:
#            itr_list[i].append((totalWaitTime[i] * 1.0 / numberOfTickets[i])
#                           - (totalPredictedTime[i] * 1.0 / numberOfTickets[i]))
#
#avgErr = [0] * 90
#stdErr = [0] * 90        
#for l in range(0,90):
#    stdErr[l] = np.std(np.array(itr_list[l]))
#    avgErr[l] = np.mean(np.array(itr_list[l]))
#dayOfWeek = range(0,90)
## print avgErr
## print stdErr
#
#plt.errorbar(dayOfWeek, avgErr, stdErr, linestyle='None', marker='^')
#plt.show()
#
#
## In[ ]:
#
#
#
#
## In[ ]:
#
## Graphs to explore aspects of the data
#
#
## In[ ]:
#
## Plot time it takes to resolve a ticket
#itr_list = [list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(),list()
#            ,list(),list(),list(),list(), list()
#            ,list(),list(),list(),list(), list() 
#           ]
#
#
#
#numberOfTickets = [0] * 90
#for line in Data100:
#    if line['time_to_resolve'] == None or int(line['time_to_resolve'] / (60*1000)) >= 90:
#        continue
#    numberOfTickets[int(line['time_to_resolve'] / (60*1000))] += 1
#    
#plt.errorbar(range(0,90), numberOfTickets)
#plt.show()
#
#
## In[ ]:
#
#resTime = list()
#for line in allData:
#    if line['time_to_resolve'] != None:
#        resTime.append(line['time_to_resolve'])
#        
#print np.std(np.array(resTime)) /(60*1000)
#print np.mean(np.array(resTime)) /(60*1000)
#
#
## In[ ]:
#
#
#
#
## In[ ]:

# Methods for getting features, labels, predictions, and models


# In[16]:

def Course_Array(course):
    code = int(re.search(r'\d+', course).group())
    return {
        8: [1,0,0,0,0,0],
        11: [0,1,0,0,0,0],
        12: [0,0,1,0,0,0],
        30: [0,0,0,1,0,0],
        100: [0,0,0,0,1,0]
    }.get(code, [0,0,0,0,0,1])

def Course_Idx(course):
    code = int(re.search(r'\d+', course).group())
    return {
        8: 0,
        11: 1,
        12: 2,
        30: 3,
        100: 4
    }.get(code, 5)


# In[17]:

def getFeatures(data):
    NUMBER_OF_COURSES = 6
    features = list()
    for line in data:
        
    # Get the estimated time per ticket -----------------------------------------------------------
        position = line['position']
        num_tutors = len(line['tutors_on_duty'])
        recent_ticket_time = line['avg_daily_res_time'] * 1.0 / (60*1000)
        if num_tutors == 0:
            num_tutors = 1
        sumVal = 0
        count = 0
        for tutor in line['tutors_on_duty']:
            if tutor in tutorStats:
                sumVal += tutorStats[tutor]
                count += 1
        if count > 0:
            tutor_bias = sumVal * 1.0 / count
        else:
            tutor_bias = 1.0
        
        time_per_ticket = [position * tutor_bias * recent_ticket_time / num_tutors] * NUMBER_OF_COURSES
        time_per_ticket_c = [a*b for a,b in zip(time_per_ticket, Course_Array(line['course']))]
    # -----------------------------------------------------------------------------------------------
    
    # Get position ----------------------------------------------------------------------------------
        position = [int(position)] * NUMBER_OF_COURSES
        position_c = [a*b for a,b in zip(position, Course_Array(line['course']))]
    # -----------------------------------------------------------------------------------------------
    
    # Get Number of Tutors --------------------------------------------------------------------------
        num_tutors = [num_tutors] * NUMBER_OF_COURSES
        num_tutors_c = [a*b for a,b in zip(num_tutors, Course_Array(line['course']))]
    # -----------------------------------------------------------------------------------------------
    
    # Get Day of the Week ---------------------------------------------------------------------------
        day_of_week_c = [0] * 7 * NUMBER_OF_COURSES
        day = int(line['day_of_week'])
        course = Course_Idx(line['course'])
        day_of_week_c[(7 * course) + day] = 1
    # -----------------------------------------------------------------------------------------------
    
    # If daily ticket count is greater than 5 then avg time to resolve for today --------------------
        
            
        features.append(time_per_ticket_c + position_c + num_tutors_c + day_of_week_c + [1])

    return features

def getFeaturesNN(data):
    features = list()
    for line in data:
        
        #position z-scored
        position = [(line['position']-5) * 1.0 / 10]
        
        #recent_ticket_time z-scored
        recent_ticket_time = [(line['avg_daily_res_time'] * 1.0 / (60*1000) - 12) * 1.0 / 15]

        #Number of tutors z-scored
        num_tutors = len(line['tutors_on_duty'])
        if num_tutors == 0:
            num_tutors = 1
        num_tutors = [(num_tutors - 3) * 1.0 / 3]
        
        #Tutor Bias (no need for z-scoring)
        sumVal = 0
        count = 0
        for tutor in line['tutors_on_duty']:
            if tutor in tutorStats:
                sumVal += tutorStats[tutor]
                count += 1
        if count > 0:
            tutor_bias = sumVal * 1.0 / count
        else:
            tutor_bias = 1.0
        tutor_bias = [tutor_bias]

        # Day of Week one hot
        day_of_week = [0] * 7
        day = int(line['day_of_week'])
        day_of_week[day] = 1
        
        # Course one hot
        course = [0] * 6
        courseIdx = Course_Idx(line['course'])
        course[courseIdx] = 1
           
            
        features.append( position + num_tutors + tutor_bias + recent_ticket_time + course + [1])

    return features

def getBasicFeatures(data):
    features = list()
    for line in data:
        position = line['position']
        num_tutors = len(line['tutors_on_duty'])
        if num_tutors == 0:
            num_tutors = 1
        time_per_ticket = [position * 1.0 / num_tutors]
        
        features.append(time_per_ticket+[1])
    return features
        
def getLabels(data):
    labels = list()
    for line in data:
        labels.append(line['diff'] * 1.0 / (60 * 1000))
    return labels


# In[26]:

# Returns a multi-level perceptron that has the same number of parameters
# as the lstsq model
def GetModel():
    sgd = SGD(lr=0.0005, momentum=0.9, decay=0.001, nesterov=True)
    model = Sequential()
    model.add(Dense(3, input_dim=11, init='normal', activation='tanh'))
    model.add(Dense(2, init='normal', activation='tanh'))
    model.add(Dense(1, init='normal'))
    
    model.compile(loss='mean_squared_error',optimizer=sgd)
    return model


# In[28]:

# Takes a list of features and the least squares coefficients and returns a list of predicted wait times
def predict_lstsq(coefficients, features):
    return [sum(np.multiply(coefficients, feat)) for feat in features]

# Takes a multi level perceptron and a list of features and returns a list of predicted wait times
def predict_nn(model, features):
    features = np.array(features)
    weights = model.get_weights()
    print ' '
    print "Weights of the model"
    print weights
    print ' '
    return model.predict(features, batch_size=10)
    


# In[ ]:




# In[20]:

# Split data and train models


# In[25]:

random.shuffle(allData)
train = allData[:(len(allData)/2)]
valadation = allData[(len(allData)/2):]

features = getBasicFeatures(train)
labels = getLabels(train)

featuresNN = getFeaturesNN(train)

features_v = getBasicFeatures(valadation)
featuresNN_v = getFeaturesNN(valadation)
labels_v = getLabels(valadation)


# In[22]:

def train_nn(features, labels):
    features = np.array(features)
    labels = np.array(labels)
    model = GetModel()
    model.fit(features, labels, batch_size=10, nb_epoch=15, verbose=1
             , shuffle=True)
    return model


# In[23]:

theta, residuals,rank,s = np.linalg.lstsq(features, labels)
model = train_nn(featuresNN ,labels)


# In[ ]:




# In[ ]:

# Exploring how the models work


# In[ ]:

# Print out theta weights
#for idx, weight in enumerate(theta):
#    if idx % 6 == 0:
#        print ' '
#    if idx < 6:
#        print str(idx % 6) + ' its wait time is ' + str(weight)
#    if idx >= 6 and idx < 12:
#        print str(idx % 6) + ' its position is ' + str(weight)
#    if idx >= 12 and idx < 18:
#        print str(idx % 6) + ' its num_tutors is ' + str(weight)
#

# In[24]:

# Prints out the weights of the nn model
#weights = model.get_weights()
#
#print weights


# In[ ]:




# In[ ]:

# Evaluating models


# In[29]:

predNN = predict_nn(model, featuresNN_v)
#pred = predict_lstsq(theta, features_v)


# In[36]:

# Cal MAE for Task3
def calc_MSE(predicted, labels):
    err = [int(abs(pred - label)) for pred, label in zip(predicted, labels) if label < 120 ]
    errsq = [val**2 for val in err]

    count = [0]*(max(err)+1)
    for ticket in err:
        count[ticket] += 1
                        
   # plt.scatter(range(0,max(err)+1), count)
    #plt.show()
    
    print "Mean Squared Error = " + str(np.mean(errsq))
    print "Standard Deviation = " + str(np.std(errsq))


# In[37]:

def calc_BinError(predicted, labels):
    offByALittle = 0
    offByALot = 0
    total = 0
    for idx in range(0,len(labels)):
        total += 1
        label = labels[idx]
        pred = predicted[idx]
        if label <= 10:
            if abs(pred - label) > 20:
                offByALot += 1
            if abs(pred - label) > 10:
                offByALittle += 1
        elif label <= 30:
            if abs(pred - label) > 20:
                offByALot += 1
            if abs(pred - label) > 10:
                offByALittle += 1
        elif label <= 60:
            if abs(pred - label) > 30:
                offByALot += 1
            if abs(pred - label) > 15:
                offByALittle += 1
        elif label <= 120:
            if abs(pred - label) > 60:
                offByALot += 1
            if abs(pred - label) > 30:
                offByALittle += 1
    print "Percent of predictions that are off by a little: " + str(offByALittle * 1.0 / total)
    print "Percent of predictions that are off by a lot: " + str(offByALot * 1.0 / total)
    print "Total number of predictions: " + str(total)


# In[38]:

#print "Performace of lstsq model"
#print "-------------------------"
#calc_BinError(pred, labels_v)
#calc_MSE(pred, labels_v)
#print ' '
print "Performace of nn model"
print "-------------------------"
calc_BinError(predNN, labels_v)
calc_MSE(predNN, labels_v)


# In[ ]:



