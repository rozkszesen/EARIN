'''
Laura Ploch, 300176, 01143517@pw.edu.pl
Julia Czmut, 300168, 01143509@pw.edu.pl
Task 4. 
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm


class Packet:
    def __init__(self) -> None:
        self.attributes = []
    

class Packets:
    def __init__(self):
        self.packets = []
    

    def load_packets_from_file(self, file_name):
        with open(file_name) as input_file:
            lines = []
            lines = input_file.read().split('\n')
            for line in lines:
                line.strip('\n')

        # one-hot encoding for protocol type, service and flag attributes
        encoder1 = OneHotEncoder(sparse=False)
        encoder2 = OneHotEncoder(sparse=False)
        encoder3 = OneHotEncoder(sparse=False)
        
        protocol_types = np.array(['tcp','udp', 'icmp'])
        service_types = np.array(['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'])
        flag_types = np.array(['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'])
        
        protocol_types = protocol_types.reshape(len(protocol_types), 1)
        service_types = service_types.reshape(len(service_types), 1)
        flag_types = flag_types.reshape(len(flag_types), 1)
        
        encoder1.fit(protocol_types)
        encoder2.fit(service_types)
        encoder3.fit(flag_types)

        for line in tqdm(lines): 

            if line == '':  # if end of file
                break

            # initialize a new Packet object and save its attributes
            new_packet = Packet()
            raw_attributes = line.split(',')
            attributes = []
            attributes.append(int(raw_attributes[0]))   # duration
            
            # one-hot encoding for protocol type attribute
            protocol_type = raw_attributes[1]   
            protocol_type = (encoder1.transform(np.array(protocol_type).reshape(1, -1))).tolist()
            protocol_type = protocol_type[0]    # unfold unnecessary dimension
            for attr in range(len(protocol_type)):
                attributes.append(protocol_type[attr])

            # one-hot encoding for service attribute
            service = raw_attributes[2]
            service = (encoder2.transform(np.array(service).reshape(1, -1))).tolist()
            service = service[0]    # unfold unnecessary dimension
            for attr in range(len(service)):
                attributes.append(service[attr])

            # one-hot encoding for flag attribute
            flag = raw_attributes[3]
            flag = (encoder3.transform(np.array(flag).reshape(1, -1))).tolist()
            flag = flag[0]    # unfold unnecessary dimension
            for attr in range(len(flag)):
                attributes.append(flag[attr])

            attributes.append(int(raw_attributes[4]))   # src_bytes
            attributes.append(int(raw_attributes[5]))   # dst_bytes
            attributes.append(int(raw_attributes[6]))   # land
            attributes.append(int(raw_attributes[7]))   # wrong_fragment
            attributes.append(int(raw_attributes[8]))   # urgent
            attributes.append(int(raw_attributes[9]))   # hot
            attributes.append(int(raw_attributes[10]))  # num_failed_logins
            attributes.append(int(raw_attributes[11]))  # logged_in
            attributes.append(int(raw_attributes[12]))   # num_compromised
            attributes.append(int(raw_attributes[13]))   # root_shell
            attributes.append(int(raw_attributes[14]))   # su_attempted
            attributes.append(int(raw_attributes[15]))   # num_root
            attributes.append(int(raw_attributes[16]))   # num_file_creations
            attributes.append(int(raw_attributes[17]))   # num_shells
            attributes.append(int(raw_attributes[18]))   # num_access_files
            attributes.append(int(raw_attributes[19]))   # num_outbound_cmds
            attributes.append(int(raw_attributes[20]))   # is_host_login
            attributes.append(int(raw_attributes[21]))   # is_guest_login
            attributes.append(int(raw_attributes[22]))   # count
            attributes.append(int(raw_attributes[23]))   # srv_count
            attributes.append(float(raw_attributes[24]))   # serror_rate
            attributes.append(float(raw_attributes[25]))   # srv_serror_rate
            attributes.append(float(raw_attributes[26]))   # rerror_rate
            attributes.append(float(raw_attributes[27]))   # srv_serror_rate
            attributes.append(float(raw_attributes[28]))   # same_srv_rate
            attributes.append(float(raw_attributes[29]))   # diff_srv_rate
            attributes.append(float(raw_attributes[30]))   # srv_diff_host_rate
            attributes.append(int(raw_attributes[31]))    # dst_host_count
            attributes.append(int(raw_attributes[32]))    # dst_host_srv_count
            attributes.append(float(raw_attributes[33]))    # dst_host_same_srv_rate
            attributes.append(float(raw_attributes[34]))    # dst_host_diff_srv_rate
            attributes.append(float(raw_attributes[35]))    # dst_host_same_src_port_rate
            attributes.append(float(raw_attributes[36]))    # dst_host_srv_diff_host_rate
            attributes.append(float(raw_attributes[37]))    # dst_host_serror_rate
            attributes.append(float(raw_attributes[38]))    # dst_host_srv_serror_rate
            attributes.append(float(raw_attributes[39]))    # dst_host_rerror_rate
            attributes.append(float(raw_attributes[40]))    # dst_host_srv_rerror_rate
            if raw_attributes[41] == 'normal':  # class
                attributes.append(int('0'))
            else:
                attributes.append(int('1'))

            # add processed attributes
            for attribute in attributes:
                new_packet.attributes.append(attribute)

            self.packets.append(new_packet)

        self.packets = self.get_xy(self.packets)
        
    def get_xy(self, packets):
        # X_train (features) - packet attributes (except from 'class')
        # Y_train (labels) - packet's 'class' attribute to predict

        X_train = []
        Y_train = []

        for packet in range(len(packets)):
            X_train.append(packets[packet].attributes[0:121])
            Y_train.append(packets[packet].attributes[122])

        return [X_train, Y_train]



if __name__ == '__main__':

    training_set = Packets()
    print("Loading training data from input file...")
    training_set.load_packets_from_file('./KDDTrain+.txt')

    test_set = Packets()
    print("Loading test data from input file...")
    test_set.load_packets_from_file('./KDDTest+.txt')

    # initialize the model
    model_decisiontree = DecisionTreeClassifier(random_state=0)
    model_kneighbors = KNeighborsClassifier(n_neighbors=10)

    # training data
    x_train = training_set.packets[0] # features
    y_train = training_set.packets[1] # labels

    # test data
    x_test = test_set.packets[0]
    y_test = test_set.packets[1]

    
    # Hyperparameter optimization (now commented out, because it was executed in Google Collab)
    '''# DecisionTree optimization:
    param_grid_decisiontree = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2, 4, 6, 8, 10, 12]}
    hgs_decisiontree = HalvingGridSearchCV(DecisionTreeClassifier(), param_grid_decisiontree)
    hgs_decisiontree.fit(x_train, y_train)
   
    print('Best criterion:', hgs_decisiontree.best_estimator_.get_params()['criterion'])
    print('Best splitter:', hgs_decisiontree.best_estimator_.get_params()['splitter'])
    print('Best max_depth:', hgs_decisiontree.best_estimator_.get_params()['max_depth'])
    
    # K-neighbors optimization
    param_grid_kneighbors = {'n_neighbors': list(range(1,30)), 'p': [1,2]}
    rs_kneighbors = RandomizedSearchCV(KNeighborsClassifier(), param_grid_kneighbors)
    rs_kneighbors.fit(x_train, y_train)

    print('Best leaf_size:', rs_kneighbors.best_estimator_.get_params()['leaf_size'])
    print('Best p:', rs_kneighbors.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', rs_kneighbors.best_estimator_.get_params()['n_neighbors'])'''

    # fit the model to our training data
    predictions_decisiontree = model_decisiontree.fit(x_train, y_train).predict(x_test)
    predictions_kneighbors = model_kneighbors.fit(x_train, y_train).predict(x_test)

    print("_____Decision Tree model_____")
    accuracy_2 = model_decisiontree.score(x_test, y_test)
    print("Accuracy: " + str(accuracy_2))

    print("_____K-nearest Neighbors model_____")
    accuracy_3 = model_kneighbors.score(x_test, y_test)
    print("Accuracy: " + str(accuracy_3))


    # plotting the results (now commented out, because it was executed in Google Collab)
    '''pred_probs = model_decisiontree.predict_proba(x_test)[:,1]
    display = PrecisionRecallDisplay.from_predictions(y_test, pred_probs)
    plt.show()

    pred_probs = model_kneighbors.predict_proba(x_test)[:,1]
    display = PrecisionRecallDisplay.from_predictions(y_test, pred_probs)
    plt.show()

    plot_confusion_matrix(model_decisiontree, x_test, y_test)
    plt.show()

    plot_confusion_matrix(model_kneighbors, x_test, y_test)
    plt.show()

    print(classification_report(y_test, predictions_kneighbors, target_names=['normal', 'anomaly']))

    print(classification_report(y_test, predictions_decisiontree, target_names=['normal', 'anomaly']))'''
    


