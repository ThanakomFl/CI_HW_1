#!/usr/bin/env python
# coding: utf-8

import numpy as np

def load_txt(path):
    f=open(path, "r")
    if(path[-3:] == 'txt'):
        contents =f.readlines()

        dataset = np.zeros((len(contents)-2,len(contents[2].split('\t'))-1))
        label = np.zeros((len(contents)-2))

        for i in range(len(contents)-2):
            x = contents[i+2].split("\t")
            for j in range(len(x)):
                if j != len(x) - 1 :
                    dataset[i][j] = float(x[j])
                else :
                    label[i] = float(x[j][:-1])
    else :
        contents =f.readlines()
        n_data = int(len(contents)/3)
        dataset = np.zeros((n_data,2))
        label = np.zeros((n_data,2))
        j = 0
        count = 0
        for i in range(len(contents)):
            if(j == 1):
                dataset[count][0] = float(contents[i].split()[0])
                dataset[count][1] = float(contents[i].split()[1])
            if(j == 2):
                label[count][0] = int(contents[i].split()[0])
                label[count][1] = int(contents[i].split()[1])
                count = count + 1
                j = -1
            j += 1
            
    return dataset,label

def norm(data_r):
    data = data_r.copy()
    if data.ndim != 1:
        maxx = [0]*len(data[0])
        minn = [9999]*len(data[0])

        for i in range(len(data[0])):
            for j in range(len(data)):
                if(data[j][i] > maxx[i]):
                    maxx[i] = data[j][i]
                if(data[j][i] < minn[i]):
                    minn[i] = data[j][i]

        for i in range(len(data[0])):
            for j in range(len(data)):
                data[j][i] = (data[j][i]-minn[i])/(maxx[i]-minn[i])
    else :
        maxx = 0
        minn = 0
        for j in range(len(data)):
            if(data[j] > maxx):
                maxx = data[j]
            if(data[j] < minn):
                minn = data[j]
        for j in range(len(data)):
            data[j] = (data[j]-minn)/(maxx-minn)  
    return data,maxx,minn

def convert_norm(pred,mx,mn):
    return pred*(mx - mn) + mn

import numpy as np
class NN :
    
    def __init__(self,shape,nueral_shape,acti_funct):
        shape[1:1] = nueral_shape 
        self.shape = shape
        self.act_func = acti_funct
        self.weights = self.init_weights(self.shape)
        self.outputs = None
        self.deltas = None
        self.del_old_weights = None
        
    def init_old_weights(self,network_shape):
        weight_arrays = []
        for i in range(0, len(network_shape) - 1):
            cur_idx = i
            next_idx = i + 1
            weight_array = np.zeros((network_shape[next_idx], network_shape[cur_idx]))
            weight_arrays.append(weight_array)
        
        return weight_arrays
    
    def init_weights(self,network_shape):
        weight_arrays = []
        for i in range(0, len(network_shape) - 1):
            cur_idx = i
            next_idx = i + 1
            weight_array = 2*np.random.rand(network_shape[next_idx], network_shape[cur_idx]) - 1
            weight_arrays.append(weight_array)

        return weight_arrays


    def predict(self,sample):
        
        current_input =  (sample.copy()).T
        outputs = []
        for network_weight in self.weights:
            current_output_temp = np.dot(network_weight, current_input)
            current_output = self.acti_funct(current_output_temp)
            outputs.append(current_output)
            current_input = current_output
        
        if(self.shape[-1] == 1) :
            return current_output.T
        else :
            tp = None 
            fp = None 
            for i in range(len(outputs[-1])):
                if( i == 0) :
                    tp = outputs[-1][i]
                else :
                    fp = outputs[-1][i]
                    tp = np.vstack((tp, fp)).T
            return np.argmax(tp, axis=1)
                    
    def train(self,sample, d_out, training_rate,momentum_rate,epoch,show=True):
        sample_T = (sample.copy()).T
        d_out_T = (d_out.copy()).T
        for i in range(epoch):
            self.FW_NN(sample_T)
            self.BW_NN(d_out_T)
            self.update_weights(sample_T,learning_rate,momentum_rate,i)
            sqe = self.sum_sqaure_error(self.predict(sample),d_out_T)
            if(show and i % 10 == 0):
                print('Epoch : #'+str(i)+',  Sum Square Error : '+str(sqe))
            if sqe < np.finfo(np.float32).eps :
                break
                
    def FW_NN(self,input):

        current_input = input
        outputs = []
        for w in self.weights:
            current_output_tmp = np.dot(w, current_input)
            current_output = self.acti_funct(current_output_tmp)
            outputs.append(current_output)
            current_input = current_output
        self.outputs = outputs
        
    def BW_NN(self,d_out):
        
        deltas = []
        O_error = d_out - self.outputs[len(self.outputs)-1]
        O_delta = O_error *self.derivertive_acti_funct(self.outputs[len(self.outputs)-1])
        deltas.append(O_delta)

        cur_delta = O_delta
        back_idx = len(self.outputs) - 2

        for w in self.weights[::-1][:-1]:
            hidd_error = np.dot(w.T, cur_delta)
            hidd_delta = hidd_error * self.derivertive_acti_funct(self.outputs[back_idx])
            deltas.append(hidd_delta)
            cur_delta = hidd_delta
            back_idx -= 1
            
        self.deltas = deltas
    
    def update_weights(self,sample,learning_rate,momentum_rate,count):
        index_current_weight = len(self.weights) - 1
        current_dels = []
        for d in self.deltas:
            sample_used = None
            if index_current_weight - 1 < 0:
                sample_used = sample
            else:
                sample_used = self.outputs[index_current_weight - 1]
                
            current_delta = learning_rate*np.dot(d, sample_used.T)
            
            if(count == 0) :
                self.weights[index_current_weight] +=  current_delta
            else :
                self.weights[index_current_weight] +=  momentum_rate*self.del_old_weights[index_current_weight]+ current_delta
            index_current_weight -= 1
            current_dels.insert(0, current_delta)
            
        self.del_old_weights = current_dels

    def acti_funct(self,v):
        if self.act_func == 'sigmoid' :
            return 1 / (1 + np.exp(-v))
        if self.act_func == 'tanh' :
            return np.tanh(v)
        if self.act_func == 'linear' :
            return v
        return v

    def derivertive_acti_funct(self,v):
        if self.act_func == 'sigmoid' :
            return v * (1 - v)
        if self.act_func == 'tanh' :
            return 1 - (v ** 2)
        if self.act_func == 'linear' :
            return 1
        return v

        
    def sum_sqaure_error(self,pred,real):
        real_m = real.copy()
        sums = 0
        if(real.ndim > 1) :
            tp = None 
            fp = None 
            for i in range(len(real_m)):
                if( i == 0) :
                    tp = real_m[i]
                else :
                    fp = real_m[i]
                    tp = np.vstack((tp, fp)).T
            real_m = np.argmax(tp, axis=1)
        for i in range(len(pred)):
            sums = sums + np.square(pred[i]-real_m[i])
        return sums/2
    
    def conf_matrix(self,pred,true,is_norm=False,confuse=True):
        true_m = np.zeros(len(true))
        if(true.ndim > 1) :
            for i in range(len(true)):
                true_m[i] = np.argmax(true[i], axis=0)
        if(is_norm):
            sqr_error = 0 
            print('Desired Output\t\t|\tPredict\t\t\t|\tError')
            print('-----------------------------------------------------------------------------')
            for i in range(len(true)):
                error = round(true[i] - round(pred[i][0],8),2)
                print(str(int(true[i]))+'\t\t\t|\t'+str(format(round(pred[i][0],8), '.8f'))+'\t\t|\t'+str(error))
                sqr_error = sqr_error + (error * error)
            print('-----------------------------------------------------------------------------')
            print('\t\t Mean Square Error = '+str(round(sqr_error/len(true),6)))
            print('=============================================================================')
            return round(sqr_error/len(true),6)
        else :
            print('Desired Output\t\t|\tPredict\t\t\t')
            print('------------------------------------------------')
            for i in range(len(true)):
                print(str(int(true_m[i]))+'\t\t\t|\t'+str(pred[i]))
            print('------------------------------------------------')
        if(confuse):
            print('\n\t\t Confusion Matrix')
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            for i in range(len(true)):
                if((pred[i] == 0) and ( true_m[i] == 0)):
                    TN = TN + 1 
                elif((pred[i] == 1) and ( true_m[i] == 1)):
                    TP = TP + 1
                elif((pred[i] == 1) and ( true_m[i] == 0)):
                    FP = FP + 1
                else :
                    FN = FN + 1

            print(' ---------------------------------------------- ')
            for i in range(8):
                print('|\t\t\t|\t\t\t|')
                if(i == 1):
                    print('|\t    '+str(TN)+'\t\t|\t    '+str(FP)+'\t\t|\t '+str(FP+TN))
                if(i == 3):
                    print(' ----------------------------------------------')
                if(i == 5):
                    print('|\t    '+str(FN)+'\t\t|\t    '+str(TP)+'\t\t|\t '+str(FN+TP))
            print(' ----------------------------------------------')
            print(' \t    '+str(TN+FN)+'\t\t\t    '+str(FP+TP)+'\t\t\t  '+str(TN+FP+FN+TP))
            print('')
            print('Accuracy : '+str((TN+TP)/(TN+FP+FN+TP)))
            return((TN+TP)/(TN+FP+FN+TP))

def load_data(name,cross):
    is_norm = False
    if(name  == 1):
        dataset,label = load_txt("./Flood_dataset.txt")
        dataset,mx_dataset,mn_dataset = norm(dataset)
        label,mx_label,mn_label = norm(label)
        is_norm = True
        max_min = [mx_dataset,mn_dataset,mx_label,mn_label]
    else :
        dataset,label = load_txt("./cross.pat")
    n_sample = np.arange(len(dataset))
    np.random.shuffle(n_sample)
    if(is_norm) :
        return dataset,label,n_sample,max_min
    else :
        return dataset,label,n_sample


def MLP(Neural,learning_rate,momentum_rate,activation,epoch,cross_valda_train_test,data_num) :
    if(data_num == 0):
        print('------------------- Variable -------------------\n')
        dataset,label,n_sample = load_data(data_num,cross_valda_train_test)
        data_name = 'cross.pat'
    else :
        print('\n-------------------------------- Variable --------------------------------\n')
        dataset,label,n_sample,max_min = load_data(data_num,cross_valda_train_test)
        data_name = 'Flood data set'
    n_test_per_round = int(len(dataset)*cross_valda_train_test[1]/100)
    print('Datafile : ' +str(data_name),end='\n')
    print('Neural name : '+str(len(dataset[0]))+'-',end='')
    for i in range(len(Neural)):
        print(str(Neural[i])+'-',end='')
    print(label.ndim,end='\n')
    print('Learning rate : '+str(learning_rate),end='\n')
    print('Momentum rate : '+str(momentum_rate),end='\n')
    print('Activaion Function : ' +str(activation),end='\n')
    print('Cross validation : ['+str(cross_valda_train_test[0])+' : '+str(cross_valda_train_test[1])+']',end='\n')
    print('#Epoch : '+str(epoch),end='\n')
    error_avg = []
    acc_avg = []
    for i in range(10):
        test_data = n_sample[i*n_test_per_round:i*n_test_per_round+n_test_per_round]
        train_data = list(set(n_sample) - set(test_data))
        nn = NN([len(dataset[0]),label.ndim],Neural,activation)
        nn.train(dataset[train_data],label[train_data],learning_rate,momentum_rate,epoch,False)
        pred = nn.predict(dataset[test_data])
        if(data_num == 1):
            print('\n-------------------------------- Round : '+str(i)+' --------------------------------')
            pred = convert_norm(pred,max_min[2],max_min[3])
            test_label = convert_norm(label[test_data],max_min[2],max_min[3])
            error_avg.append(nn.conf_matrix(pred,test_label,is_norm=True,confuse=False))
        else :
            print('\n----------------- Round : '+str(i)+' -----------------')
            acc_avg.append(nn.conf_matrix(pred,label[test_data],is_norm=False,confuse=True))
    if(data_num == 1):
        print('\n********  Mean Square Error Average : ' + str(round(np.sum(error_avg)/len(error_avg),4))+'  *********')
    else :
        print('\n********  Accuracy Average : ' + str(round(np.sum(acc_avg)/len(acc_avg),4))+'  *********')


Neural = [10,6] 
cross_valda_train_test = [90,10] # train 90 , test 10
data_num = 1 # 0 = cross.pat , 1 = flood data set
learning_rate = 0.12
momentum_rate = 0.07
activation = 'sigmoid'
epoch = 1000
MLP(Neural,learning_rate,momentum_rate,activation,epoch,cross_valda_train_test,data_num)

