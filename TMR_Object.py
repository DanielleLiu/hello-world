import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as mins


class MemoryTrace():
    def __init__(self,encode_str,encode_t):
        self.initial_str=encode_str
        self.create_time=encode_t  # note, assumes new memory traces, would be different for re-activating semantic traces like words
        self.forgetting=0.0        # decay might have to be a list if it changes over time
        self.events=[]

    def add_event(self,encode_str,encode_t): # note, will be for TMR, re-study and retrieval events eventually
        self.events.append([encode_str,encode_t])
        return

    # this method gets strength at a single point in time taking into account all previous boost events
    def strength_t(self,t):
        if len(self.events)==0 or self.events[0][1] > t:
            since=t-self.create_time
            if since<0:
               return 0.0
            else:
                return self.initial_str*np.exp(self.forgetting*since)
        else:
            y = self.initial_str*np.exp(self.forgetting*(self.events[0][1]-self.create_time))
            y = (1-((1-y)/(1+self.events[0][0]))) # strength right after first boost
            i = 0 #incase only 1 tmr events, index =0
            for i in range(1,len(self.events),1):
               if t>self.events[i][1]:
                   y = y*np.exp(self.forgetting*(self.events[i][1]-self.events[i-1][1])) #current strength
                   y = (1-((1-y)/(1+self.events[i][0]))) # new y
               else: break
            if  (t- self.events[i][1]) > 0: # last added events i; if t inbtw 2 events or after last events; y will decay in the remaining interval
                y = y*np.exp(self.forgetting*(t-self.events[i][1]))
            return y

    # generate strength across a list of time points
    def strength(self,time_range):
        y=[]
        for i in time_range:
            y.append(self.strength_t(i))
        return np.array(y)

    def graph(self,time_range,legend='Default legend',color='blue'):
        t=np.arange(time_range[0],time_range[1])# step is optional parameter, for better accuracy, step should be <1
        y=self.strength(t)
        plt.plot(t,y,label=legend,color=color)

    # calculate the decay parameter, set to negative
    def find_decay(self,test_score,test_time):
        str_change=test_score/self.initial_str  # ratio decay
        elap=self.create_time-test_time
        self.forgetting=-np.log(str_change)/elap
        return

    #add a cueing/study events to the events list
    def find_tmr(self,tmr_time,test_time,test_score):  # estimate the strength of the tmr event that produced the test
        gap=test_score-self.strength_t(test_time) #actual post strenth minus if excluding current boost event, at post test time strength
        tmr=gap/np.exp(self.forgetting*(test_time-tmr_time))
        self.add_event(tmr,tmr_time)
        return tmr

    # find tmr assuming multiplication together
    # parameters: tmr_times=list of cueing timestamps in min, t_pre,t_post,
    # pre=pre strength either predicted from regression or actual, post=post strength actual;
    def find_multi_tmr2(self,tmr_times,t_post,post):
        phi=(3-5**0.5)/2
        def fcn(tmr):
            self.events=[] #initialize events everytime
            for i in tmr_times:
                self.add_event(tmr,i)
            return abs(self.strength_t(t_post) - post) #error in absolute value

        def goldSection(a,b,c,runs,precision = 0.0001): #always a<b<c
            runs +=1
            if abs(c-a) < precision:
                return b,fcn(b),runs
            else:
                if (b-a)>(c-b): #ab is larger segment; new = axb or xbc
                    x = b - phi * (b-a)
                    if fcn(x) > fcn (b): # min at b, xbc
                        return goldSection(x,b,c,runs,precision)
                    else: # min at x, axb
                        return goldSection(a,x,b,runs,precision)
                else: # bc is larger segment; new = abx, or bxc
                    x = b + phi * (c-b)
                    if fcn(x) > fcn (b): # min at b, abx
                        return goldSection(a,b,x,runs,precision)
                    else: # min at x, bxc
                        return goldSection(b,x,c,runs,precision)
        tmr,minError,runs = goldSection(0,500,10000,0,0.0001)
        #print "x (tmr boost tried),minError,runs",tmr,minError,runs
        #print "tmr,minError,run times",tmr,minError,runs
        # check if self.events have everything
        return tmr

    def traceVector(self,time_range=[-5,300],step=0.1):
        t=np.arange(time_range[0],time_range[1],step)# step is optional parameter, for better accuracy, step should be <1
        y=self.strength(t)
        return t,np.array(y)

def estimate_multi_tmr(c1_pre,c1_post,c2_pre,c2_post,t1_pre=0.0,t1_post=90.0,t2_pre=0.0,t2_post=90.0,tmr_ts=[],c1_label='Uncued',c2_label='Cued',graph=False,c1_color='blue',c2_color='red'):
    C1=MemoryTrace(c1_pre,t1_pre) #initialize a memorytrace object
    C1.find_decay(c1_post,t1_post) #set self.forgetting=decay param=log(postMemStrength/pre)/deltaTime

    if graph:
        time_range=[t1_pre,t1_post+30]
        C1.graph(time_range,legend=c1_label,color=c1_color)
        #plot the uncued pre/post points
        plt.plot(t1_pre,c1_pre,marker='o',color=c1_color)
        plt.plot(t1_post,c1_post,marker='o',color=c1_color)

    C2=MemoryTrace(c2_pre,t2_pre) #initialize memorytrace for uncued
    C2.forgetting=C1.forgetting #set same decay parameter

    # TMR event, assume happens at half nap time = dealy/2=45
    tmr=C2.find_multi_tmr2(tmr_ts,t2_post,c2_post)
    print "TMR event:",C2.events

    if graph:
        C2.graph([t2_pre,t2_post+30],legend=c2_label,color=c2_color)
        plt.plot(t2_pre,c2_pre,marker='o',color=c2_color)
        plt.plot(t2_post,(c2_post),marker='o',color=c2_color)
        # for i in C2.events:
        #     plt.plot(i[1],C2.strength_t(i[1]),marker='o',color=c2_color)
        plt.ylabel('Memory Strength\n Strength=1-error in pixels/300')
        plt.xlabel('Time (min)')
        plt.legend(loc='best')

    # TMR relative strength
    init_encode=(C1.initial_str+C2.initial_str)/2.0
    #if graph:
        #print "Initial avg strength %.2f, boost per tmr %.2f, Percentage (one tmr boost/avg strength) %.3f" % (init_encode,tmr,tmr/init_encode)
    return (init_encode,tmr,C1.forgetting)

def estimate_tmr(c1_pre,c1_post,c2_pre,c2_post,t1_pre=0.0,t1_post=90.0,t2_pre=0.0,t2_post=90.0,c1_label='Uncued',c2_label='Cued',graph=False,c1_color='blue',c2_color='red',t0=-5):
    time_range=[t0,t1_post+30]
    C1=MemoryTrace(c1_pre,t1_pre) #initialize a memorytrace object
    C1.find_decay(c1_post,t1_post) #set self.forgetting=decay param=log(postMemStrength/pre)/deltaTime

    if graph:
        #plot a roughly mooth line connecting t from -5 to delay+30, step1; y=strength across all t (smooth line connecting all points)
        C1.graph(time_range,legend=c1_label,color=c1_color)
        #plot the uncued pre/post points
        plt.plot(t1_pre,c1_pre,marker='o',color=c1_color)
        plt.plot(t1_post,c1_post,marker='o',color=c1_color)

    C2=MemoryTrace(c2_pre,t2_pre) #initialize memorytrace for uncued
    C2.forgetting=C1.forgetting #set same decay parameter

    # TMR event, assume happens at half nap time = dealy/2=45
    tmr=C2.find_tmr(t2_pre+45.0,t2_post,c2_post)

    if graph:
        #time_range_tmr=[-5,t2_post+30]
        C2.graph(time_range,legend=c2_label,color=c2_color)
        plt.plot(t2_pre,c2_pre,marker='o',color=c2_color)
        plt.plot(t2_post,(c2_post),marker='o',color=c2_color)

        plt.ylabel('Memory Strength\n Strength=1-error in pixels/300')
        plt.xlabel('Time (min)')
        plt.legend(loc='best')

    # TMR relative strength
    init_encode=(C1.initial_str+C2.initial_str)/2.0
    #if graph:
        #print "Initial avg strength %.2f, TMR boost %.2f, Percentage %.3f" % (init_encode,tmr,tmr/init_encode)
        #initial avg=average bewteen cued+uncued all subjects average
    return (init_encode,tmr,C1.forgetting)

# add a new MemoryTrace with given decay parameter
def add_tmr(pre,post,decay,t_pre,t_post,label='data',color='green',t0=-5):

    time_range=[t0,t_post+30]
    C=MemoryTrace(pre,t_pre)
    C.forgetting=decay

    # TMR event: assume 45min after pretest
    tmr=C.find_tmr(45+t_pre,t_post,post)
    #print "Add_tmr value: ",tmr

    C.graph(time_range,legend=label,color=color)

    plt.plot(t_pre,pre,marker='o',color=color)
    plt.plot(t_post,post,marker='o',color=color)
    plt.legend(loc='best')
    return tmr


def individs(exp_data):
    # estimates for each participant
    tmr_mat=[]
    for i in range(exp_data.shape[0]):
        (avg_init,tmr,decay)=estimate_tmr(exp_data[i,1],exp_data[i,2],exp_data[i,4],exp_data[i,5])
        tmr_mat.append([exp_data[i,0],avg_init,tmr,
            pixels_to_str(exp_data[i,1]),pixels_to_str(exp_data[i,4]),
            pixels_to_str(exp_data[i,2]),pixels_to_str(exp_data[i,5]),
            decay])
    tm=np.array(tmr_mat)
    # column order: subj_num, initial strength (average of cued and uncued initial strength), tmr,
    # uncued_pre in strength, cued_pre, uncued_post,uncued_post,decay(post-pretest time)

    tm_filt=tm[tm[:,2]<1.0,:]  # exclude excessive TMR values, >1.0 (100% of initial strength); TMR=iboost in strength, should <=1

    # this should be an output to csv function
    for i in range(tm_filt.shape[0]):
        print "%d,%.2f,%.3f,%.2f,%.2f,%.2f,%.2f,%.3f" % (tm_filt[i,0],tm_filt[i,1],tm_filt[i,2],
            tm_filt[i,3],tm_filt[i,4],tm_filt[i,5],tm_filt[i,6],tm_filt[i,7])

    # todo: load all-cued/spindle dataset and compare
    # formalize graphs of relationshoips between tmr estiamte and initial encoding strength
    # define outlier process, e.g., S19 in Exp1 (forgets to zero for uncued, leading to massive overestimation of cuing effect)

    tm_means=np.mean(tm_filt[:,1:],axis=0) #average of all subjects' averages
    tm_se=np.std(tm_filt[:,1:],ddof=1,axis=0)/np.sqrt(tm_filt.shape[0])
    # bar plot of average initial_strength, tmr of all subjects' average
    plt.figure()
    plt.subplot(2,2,1)
    plt.bar(np.arange(2),tm_means[:2],0.50,yerr=tm_se[:2],
        color=('b','r'))
    plt.xticks(np.arange(2)+0.25,("Init","TMR"))
    plt.axis([-0.5,2.0,0,0.6])
    plt.ylabel("Memory Strength")

    # bar plot of pre_cued, uncued, post_cued, uncued of subjects' average and std
    plt.subplot(2,2,2)
    plt.bar(np.arange(4),tm_means[2:6],0.50,yerr=tm_se[2:6],color=('b','r','b','r'))
    plt.xticks(np.arange(4)+0.25,("Pre-U","Pre-C","Post-U","Post-C"))
    plt.axis([-0.5,4.0,0.25,0.75])
    plt.ylabel("Memory Strength")

    # scatter plot/points plot of average pre-post initial_strength and tmr effects
    plt.subplot(2,2,3)
    plt.plot(tm_filt[:,2],tm_filt[:,1],'.')
    plt.axis([-0.25,0.5,0.25,0.75])
    plt.xlabel("TMR Effect")
    plt.ylabel("Initial Strength")

    # points plot of cued_pre_memory strength vs tmr effects
    plt.subplot(2,2,4)
    plt.plot(tm_filt[:,2],tm_filt[:,4],'.')
    plt.axis([-0.25,0.5,0.25,0.75])
    plt.xlabel("TMR Effect")
    plt.ylabel("Initial Strength (Cued)")

    plt.show()
    return

# this function converts pixel error to strength
def pixels_to_str(pixels):
    return max(0.001,1.0-pixels/300)   # error > 300, assume str=0.001 almost zero

# Example Usage of the Class:
# def data12Plot():
#     (strength,tmr_size,decay)=estimate_tmr(data_means[1],data_means[2],data_means[4],data_means[5],delay=240,
#         c1_label='Uncued',c2_label='Cued',graph=True) #pass: pre,post(uncued),pre,post(cued)
#     print "Decay: ",decay,"\n"
#     add_tmr(data_means2[1],data_means2[2],240,decay,label='Cued-spindle',color='yellow')
#     add_tmr(data_means2[4],data_means2[5],240,decay,label='Cued+spindle',color='green')

