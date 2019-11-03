from datetime import datetime
from csv import DictReader, reader
import sys
import csv
from datetime import datetime
csv.field_size_limit(sys.maxsize)

def epoch2datetime(x):
    epoch = (x + 1465876799998) / 1000.
    tstp = datetime.fromtimestamp(epoch)
    return [tstp.weekday(), tstp.hour]

print("Content..")
with open("promoted_content.csv") as infile:
    prcont = reader(infile)
    #prcont_header = (prcont.next())[1:]
    prcont_header = next(prcont)[1:]
    prcont_dict = {}
    for ind,row in enumerate(prcont):
        prcont_dict[int(row[0])] = row[1:]
    print(len(prcont_dict))
del prcont

print("Events..")
with open("events.csv") as infile:
    events = reader(infile)
    #events.next()
    next(events)
    event_header = ['uuid', 'document_id', 'platform', 'geo_location', 'loc_country', 'loc_state', 'loc_dma']
    event_dict = {}
    for ind, row in enumerate(events):
        tlist = row[1:3] + row[4:5]
        time_stamp = int(row[3])
        loc = row[5].split('>')
        if len(loc) == 3:
            tlist.extend(loc[:])
        elif len(loc) == 2:
            tlist.extend( loc[:]+[''])
        elif len(loc) == 1:
            tlist.extend( loc[:]+['',''])
        else:
            tlist.append(['','',''])
        tlist.extend(epoch2datetime(time_stamp))
        event_dict[int(row[0])] = tlist[:] 
    print(len(event_dict))
del events

print("Leakage file..")
leak_uuid_dict= {}

# -1 '1 |a 42337 |b 1  |u cb8c55702adb93  |d 379743  |p 3  |c US  |s SC  |l 519  |w 1  |h 1  |x 938164  

def csv_to_vw(loc_csv, loc_output, train=True):
    start = datetime.now()
    print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))

    with open(loc_output,"wb") as outfile:
        for t, row in enumerate(DictReader(open(loc_csv))):
            disp_id = int(row['display_id'])
            ad_id = int(row['ad_id'])
            # if t >= 1:break
            #print t, row
            
            ids_features = "|a ad_%s |b disp_%s"% (row['ad_id'], row['display_id'])

            ### Promoted content
            row_content = prcont_dict.get(ad_id, [])
            # build promoted vars
            # headers: x=document_id, y=campaign_id, z=advertiser_id
            promoted_namespaces = ['x', 'y', 'z']
            promoted_features = ""
            ad_doc_id = -1
            for i,v in enumerate(row_content):
                #print i, v
                if i == 0:
                    ad_doc_id = int(v)
                promoted_features += " |%s %s_%s" % (promoted_namespaces[i], promoted_namespaces[i], v)
            #print promoted_features  

            ### Events row
            row_events = event_dict.get(disp_id, [])
            #print row_events
            # Create cat vars for events
            events_namespaces = ['u', 'd', 'p', 'c', 's', 'l', 'w', 'h']
            disp_doc_id = -1
            event_features = ""
            if len(row_events) == 0:
                for n in events_namespaces:
                    event_features += " |%s na" % (n)
                
            for i,v in enumerate(row_events):
                if len(row_events) == 0:
                    print 'null events'
                if i == 0:
                    uuid_val = v
                if i == 1:
                    disp_doc_id = int(v)
                event_features += " |%s %s_%s" % (events_namespaces[i], events_namespaces[i], v)
            #print categorical_features

            features = ids_features + event_features + promoted_features
            # print features

            # Creating the labels
            if train:
                if row['clicked'] == "1":
                    label = 1
                else:
                    label = -1
                outfile.write( "%s '%s %s\n" % (label, t+1, features ) )
            else:
                outfile.write( "1 '%s %s\n" % (t+1, features) )

            # Reporting progress
            #if t % 1000000 == 0:
                #print("%s\t%s"%(t, str(datetime.now() - start)))
                
        #print("\n %s Task execution time:\n\t%s"%(t, str(datetime.now() - start)))
        
        return row, row_content, row_events, features


csv_to_vw(loc_csv='clicks_train.csv', loc_output='train.w', train=True)
csv_to_vw(loc_csv='clicks_test.csv', loc_output='test.w', train=False)