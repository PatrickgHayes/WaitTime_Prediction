# collect data for training 
# handling the data in memory import pandas 
# data to get our data from... 
import MySQLdb 
import csv 
import pandas
import datetime

def crossesBarrier(event1, event2):
	if event1.time() < datetime.time(4,0) and event2.time() > datetime.time(4,0):
		return True
	elif (event1.date() < event2.date() and 
			(event1.time() < datetime.time(4,0) or event2.time() > datetime.time(4,0))):
		return True
	else:
		return False



db = MySQLdb.connect(host="localhost", # your host, usually localhost 
user="root", # your username 
passwd="", # your password 
db="autograder")

events = pandas.read_sql("""select TicketEvent.type, Event.timestamp, Ticket.queue_id, TicketEvent.ticket_id
						from TicketEvent, Event, Ticket 
						where TicketEvent.id = Event.id and
	 					TicketEvent.ticket_id = Ticket.id and
	 					Ticket.queue_id = 14
	 					order by Event.timestamp;""", db) 
#print events

results = pandas.DataFrame(columns=('ticket_id', 'position', "createdAt", "acceptedAt", "diff"))
counter = 0 
index = 0
queue = set()
prevEventTime = None
# loop through all the events for the entire history of this queue from oldest to newest
for i in events.index:
	tid = events["ticket_id"][i]

	if prevEventTime and crossesBarrier(prevEventTime, events['timestamp'][i]):
			queue.clear()

	if events['type'][i] == "CREATED":
		accepted = events['timestamp'][i]
		for j in range(i, len(events.index)):
			# crossed barrier
			if crossesBarrier(events['timestamp'][i], events['timestamp'][j]):
				break;
			# found accepted event
		 	if events['ticket_id'][i] == events['ticket_id'][j] and events['type'][j] == "ACCEPTED":
		 		accepted = events['timestamp'][j]
		 		break;

		index += 1

		diff = accepted - events['timestamp'][i]
		results.loc[index] = [events['ticket_id'][i], len(queue), events['timestamp'][i], accepted, diff]


	if events['type'][i] == "CREATED" or events['type'][i] == "DEFERRED":
		if not tid in queue:
			queue.add(tid)
	elif events['type'][i] == "CANCELED" or events['type'][i] == "RESOLVED":
		if tid in queue:
			queue.remove(tid)

	prevEventTime = events['timestamp'][i]

results = results[results['diff'] != pandas.Timedelta('0 days')]
results = results[results['diff'] < pandas.Timedelta('2 hours')]
print results
print results["diff"].mean()


results.to_csv("resultsAG.csv")


