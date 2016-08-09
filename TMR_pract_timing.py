import numpy as np
from scipy import stats
import glob
import matplotlib.pyplot as plt
import TMR_Object as model

# data analysis fcns: percentDiffTest(), stats test for percentage difference, pixel vs strength comparison


#todo: inherite TMR_object class and add function of adding events & calculate regression,
# return slope, intercept and the estimated start point


#pixel strength comparision, average taking as all subjects and all items
# consider btw & within subjects variation: mdiff,mrsqr as each subject's average: available file (TMRfitting/percentdiffStrength.txt")("TMRfitting/percentdiffPixels.txt")
def psComp():
    mdiff=[]
    mrsqr=[]
    mdiff=np.array(mdiff)
    mrsqr=np.array(mrsqr)
    mdiffPix=[]
    mrsqrPix=[]
    testr=[]
    fls=glob.glob("study_events12/study_[0-9][0-9].txt")
    # fout = open("data2actualY.txt",'w')
    # f1out= open ("data2predictedY.txt",'w')
    for f in fls:
        subnum = f[-6:-4]
        data = read_one_subject(f)
        errorm,timem,lminfo = lregression(data)
        lmcurr=np.array(lminfo)
        mrsqr = np.append(mrsqr,lmcurr[:,3],axis=0)
        mdiff = np.append(mdiff, lmcurr[:,6],axis=0) #3r^2,5diff, 6 percentagediff, #7ypredicted,8yactual
        testr = np.append(testr,lmcurr[:,5]) #diff
        # for i in lmcurr[:,7]:
        #     f1out.write(str(i)+"   ")
        # f1out.write('\n')
        # for i in lmcurr[:,8]:
        #     fout.write(str(i)+"    ")
        # fout.write('\n')
        print lmcurr[:,7],lmcurr[:,8]
        errorm,timem,lminfo = lregression(data,fcn="errorPixels(error)")
        lmcurr1=np.array(lminfo)
        mrsqrPix = np.append(mrsqrPix,lmcurr1[:,3])
        mdiffPix = np.append(mdiffPix, lmcurr1[:,6])
    # fout.close()
    # f1out.close()

    print np.mean(abs(mdiff),axis=0),np.mean(abs(mdiffPix),axis=0)
    print stats.ttest_rel(abs(mdiff),abs(mdiffPix))

    print np.mean(mdiff,axis=0),np.mean(mdiffPix,axis=0)
    print stats.ttest_rel((mdiff),(mdiffPix))

    print np.mean((mrsqr),axis=0),np.mean((mrsqrPix),axis=0)
    print stats.ttest_rel(mrsqr,mrsqrPix)

    print np.mean(testr,axis=0),np.std(testr,axis=0)
    print stats.ttest_1samp(testr,0.0)


#using uncued only
def data1Decays():
    Decays=[]
    fls=glob.glob("study_events12/study_[0-9][0-9].txt")
    for f in fls:
        print f
        subnum = f[-6:-4]
        raw_data = read_one_subject(f) #0sub,1pic,3uncued0/cued1/spindle2,5errorpixels,6time
        data=raw_data[raw_data[:,3]==0]
        pics=np.unique(data[:,1])
        # [0pic, 1slope, 2intercept, 3rsqr, 4len(error),5diff,
        # 6diff/float(prey_lm)*100.0,7prey_lm,8prey]
        for pic in pics:
            curr=data[data[:,1] == pic]
            error=[]
            for i in curr[:-1,5]:
                error.append(model.pixels_to_str(i))
            time = curr [:-1,6].tolist()

            fit = np.polyfit(time,error,1)
            fitfn=np.poly1d(fit)
            pret=time[-1]
            prey=error[-1]
            prey_lm=fitfn(pret)
            t_pre=time[-1]
            #print prey,prey_lm,slope*t_pre+intercept
            # uncued skeleton
            C=model.MemoryTrace(prey_lm,t_pre)
            C.find_decay(model.pixels_to_str(curr[-1,5]),curr[-1,6])#240*60000+t_pre)
            decay_lm=C.forgetting

            C1=model.MemoryTrace(prey,t_pre)
            C1.find_decay(model.pixels_to_str(curr[-1,5]),curr[-1,6])#240*60000+t_pre)#curr[-1,6])
            decay=C1.forgetting
            # print decay_lm,decay
            # print C.initial_str,C1.initial_str
            # print prey_lm,prey,'\n'

            Decays.append(map(float,[subnum,pic,decay_lm,decay,prey_lm,prey]))

            # cued skeleton
            # C2=model.MemoryTrace(model.pixels_to_str(i[7]),t_pre)
            # C2.forgetting=C.forgetting
            # tmr=C2.find_tmr(45,C2t_post,model.pixels_to_str(C2_post_acc_actual))
            # C21=model.MemoryTrace(model.pixels_to_str(i[8]),t_pre)
            # C21.forgetting=C1.forgetting
            # tmr=C21.find_tmr(45,C2t_post,model.pixels_to_str(C2_post_acc_actual))
    Decays=np.array(Decays)
   # print np.mean(Decays[:,2],axis=0)
   # print np.mean(Decays[:,3],axis=0)

    subs=np.unique(Decays[:,0])
    Dbysub=[]
    for s in subs:
        curr=Decays[Decays[:,0]==s]
        data_means=np.mean(curr,axis=0)
        Dbysub.append(map(float,[data_means[0],data_means[2],data_means[3],data_means[4],data_means[5]]))
    # Dbysub column order: sub_num,decay_lm,decay,prey_lm in strength,prey in strength
    Dbysub=np.array(Dbysub)
    bysub_means = np.mean(Dbysub,axis=0)
    print "All average in min (lm,actual):", bysub_means[1]*60000,bysub_means[2]*60000
    print np.std(Dbysub[:,1]),np.std(Dbysub[:,2])
    print stats.ttest_rel(Dbysub[:,1],Dbysub[:,2])

    print "Average in min (lm,actual)",np.mean(Decays[:,2],axis=0)*60000,np.mean(Decays[:,3],axis=0)*60000
    print stats.ttest_rel(Dbysub[:,1]*60000,Dbysub[:,2]*60000),'\n'
    #fit with avg of sub's avg; can't use same analysis approach as t different by subject & pics now

    def writeDecay():
        #column order for "data1uncuedDecay.txt": subj_num, pic_num,
        # decay parameter with pretest strength predicted by lm, decay with actual pre_acc_strength
        # column order for "data1uncuedBySubject": subj_num, subj_avg_decay with lm_y
        # subj_avg decay with actual pre_acc_strength
        f = open("data1uncuedDecayBySubject.txt",'w')
        for row in Dbysub:
            for j in row:
                f.write(str(j)+"    ")
            f.write('\n')
        f.close()
    #writeDecay()


def data3Decays(flname,readfcn='readB()'):
    def readA():
        f = open(flname)
        d = f.readlines()
        f.close()
        m=[]
        for i in d:
            line=map(float,i.split())
            # if line[0]==210:
            #     continue
            m.append(line)
        exp3A0_data=np.array(m)
        subj3A=np.unique(exp3A0_data[:,0])
        return exp3A0_data,subj3A

    # Columns: subj_num, Pre_Acc_0 (uncued), Post_Acc_0, Acc_Diff_0, objAID, pairID
    def readB():
        f = open(flname)
        d = f.readlines()
        f.close()
        m=[]
        for i in d:
            line=map(float,i.split())
            m.append(line)
        exp3A0_data=np.array(m)
        subj3A=np.unique(exp3A0_data[:,0])
        return exp3A0_data,subj3A

    exp3A0_data,subj3A=eval(readfcn)

    decays3A=[]
    for s in subj3A:
        curr=exp3A0_data[exp3A0_data[:,0] == s]
        for pic in curr:
            C1=model.MemoryTrace(model.pixels_to_str(pic[1]),0.0)
            C1.find_decay(model.pixels_to_str(pic[2]),90) #90
            decay=C1.forgetting
            decays3A.append(map(float,[s,pic[4],pic[5],decay,pic[1],pic[2]]))

    #sub_num, objAid, pairID,decay,pre_acc,post_acc
    decays3A=np.array(decays3A)
    print "All decay average:", np.mean(decays3A,axis=0)[3]

    decays3ASub=[]
    for s in np.unique(decays3A[:,0]):
        curr=decays3A[decays3A[:,0] == s]
        #print "Curr",curr
        data_means=np.mean(curr,axis=0)
        decays3ASub.append(map(float,[data_means[0],data_means[3],data_means[4],data_means[5]]))

    #column order: 0sub_num,1decay,2pre_acc,3post_acc
    decays3ASub=np.array(decays3ASub)
    Sub3A_means= np.mean(decays3ASub,axis=0)
    print "Subject Average:", Sub3A_means[1]
    print "Std of decay:",np.std(decays3A[:,3]),np.std(decays3ASub[:,1])
    # print stats.ttest_rel(Dbysub[:,1],Dbysub[:,2])
    #print decays3ASub
    #print model.pixels_to_str(decays3ASub[0,2]),model.pixels_to_str(decays3ASub[0,3])

    C1=model.MemoryTrace(model.pixels_to_str(Sub3A_means[2]),0.0)
    C1.find_decay(model.pixels_to_str(Sub3A_means[3]),90) #90
    decay=C1.forgetting
    print "Verify decay",decay

    #plt.figure()
    #model.estimate_tmr(decays3ASub[0,2],decays3ASub[0,3],0.0,0.0,90*60000,graph=True)
    #plt.show()

    def writeFile():
        f = open("data3BuncuedDecayBySubject.txt",'w')
        for row in decays3ASub:
            for j in row:
                f.write(str(j)+"    ")
            f.write('\n')
        f.close()

        f = open("data3BuncuedDecay.txt",'w')
        for row in decays3A:
            for j in row:
                f.write(str(j)+"    ")
            f.write('\n')
        f.close()


#data1Decays()
#data3Decays("/Users/mac/Desktop/MNEMONIC-memory-model/TMR_Exp3Auncued.txt")
#data3Decays("/Users/mac/Desktop/MNEMONIC-memory-model/TMR_Exp1.txt")
#data3Decays("/Users/mac/Desktop/MNEMONIC-memory-model/TMR_Exp3Buncued.txt")
#psComp()


def read_one_subject(fname):
    f= open(fname)
    d=f.readlines()
    f.close()
    m=[]
    for i in d:
        row=map(float,i.split())
        m.append(row)
    data=np.array(m)
    return data

def lregression(data,color='green',fcn='errorStrength(error)'):
    pics= np.unique(data[:,1])
    lminfo=[]
    pics=[pics[2],pics[12]]
    for pic in pics:
        curr=data[data[:,1] == pic]
        time = curr [:-1,6]
        error=[]
        def errorStrength(error):
            for i in curr[:-1,5]:
                error.append(model.pixels_to_str(i))
        def errorPixels(error):
            error += curr[:-1,5].tolist() #use pixels
        eval(fcn)

        temp=[]
        for i in error:
            print i
            temp.append(-np.log(1-i))
        error=temp

        X=  np.vstack([time, np.zeros(len(time))]).T #stack time and 1 row wise then transpose
        m = np.linalg.lstsq(X,error)[0][0]

        pret=time[-1]
        prey=error[-1]
        prey_lm=1-np.exp(-m*pret)
        print error,m*pret,"  Convert back:",prey_lm
        if fcn == 'errorStrength(error)' and prey_lm >1.0 :
            prey_lm=1.0
            print "Prediction Error", pic

        diff=prey_lm-prey
        posty=model.pixels_to_str(curr[-1,5])
        postt=curr[-1,6]
        lminfo.append([pic,m,diff,diff/prey_lm,prey_lm,prey,pret,posty,postt,curr[-1,3]])
        print "lminfo values:",[pic,m,diff,diff/prey_lm,prey_lm,prey,pret,posty,postt,curr[-1,3]]
        print "prey_lm, prey:",prey_lm,1-np.exp(-prey)

        #column order: 0pic, 1slope, 2diff, 3percentdiff, 4predicted_y,
        # 5actual_y,6pre_t,7post_y,8post_t, 9uncued0/cued1/spindle+cued2

        # other linear regression methods (intercept changable)
        # stats.linregress return: slope, intercept, correlation_coefficient, 2-sided pvalue for slope, stadard error for slope
        # regression of y=error in pixels, x=time in ms
        #(slope, intercept, r_value, p_value, std_err) = stats.linregress(time,error)
        #rsqr = r_value**2 #R-squared: % of dependent variable variation explained by the model
        # fit = np.polyfit(time,error,1)
        # print "polyfit:",fit
        # fitfn=np.poly1d(fit)
        # prey_lm=fitfn(pret)
        # diff=prey_lm-prey
        #lminfo.append([pic, slope, intercept, rsqr, len(error),diff,diff/float(prey_lm)*100.0,prey_lm,prey])
        # plt.plot(time,fitfn(time),color)

    # plt.figure()
    # plt.plot(time,error,'o'+color[0])
    # plt.plot(time,m*time,color)
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Memory Strength")
    # plt.title("Picture: "+str(pic))

    return lminfo

data=read_one_subject("study_events12/study_19.txt")
lregression(data)
# print "testing fcn call"
# lregression(data,fcn='errorPixels(error)')
# plt.show()


def writeOne(subnum,lminfo,fname="LinearRegression12/lmint0_"):
    # lm_subj column order: 0pic_num, 1slope, 2intercept, 3multiple R^2, 4number of data poitns
    # 5diff= strength at pretest time from model - actual pretest acc,
    # 6percentage difference = diff/predicted pretest acc *100,
    # 7y(strength) predicted by regression, 8actual_strength (pre_acc)

    f = open(fname+str(subnum)+".txt",'w')
    for row in lminfo:
        for j in row:
            f.write(str(j)+"    ")
        f.write('\n')
    f.close()

def writeAll():
    slopes=[]
    uncuedm=[]
    cuedm=[]
    cuedall=[]
    uncuedall=[]
    fls=glob.glob("study_events12/study_[0-9][0-9][0-9].txt")
    for f in fls:
        print f
        subnum = f[-7:-4]
        data = read_one_subject(f)
        lminfo = np.array(lregression(data))

        # cued=np.mean(lminfo[lminfo[:,9]==1],axis=0)
        # uncued=np.mean(lminfo[lminfo[:,9]==2],axis=0)
        # lm_means = np.mean(lminfo,axis=0)
        # slopes.append(lm_means[1])
        # uncuedm.append([subnum,uncued[5],uncued[4],uncued[7],uncued[6],uncued[8]])
        # cuedm.append([subnum,cued[5],cued[4],cued[7],cued[6],cued[8]])
        # cuedall.append(lminfo[lminfo[:,9]==1][:,5].tolist())
        # uncuedall.append(lminfo[lminfo[:,9]==2][:,5].tolist())

        #writeOne(subnum,lminfo,"LinearRegression12/lg_")

    # print stats.ttest_1samp(slopes,0.0) #p=5.45E-5 strength; p=1.19E-5 pixels
    # print "Average Slope:",sum(slopes)/len(slopes)
    # # avg and std of the cued, uncued subjects
    # cuedall = np.array(cuedall)
    # uncuedall = np.array(uncuedall)
    # print "Actual Cued mean:", np.mean(cuedall),np.std(cuedall)
    # print "Actual Uncued mean:",np.mean(uncuedall),np.std(uncuedall)

#writeAll()

def fitmodel():
    f=open("/Users/mac/Desktop/MNEMONIC-memory-model/TMR_Exp3ASubject.txt")
    d=f.readlines()
    f.close()
    # columns: subj_num, uncued_pre,post,diff,cued_pre, cued_post, cued_diff,
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    exp3A_data=np.array(m)

    f=open("/Users/mac/Desktop/MNEMONIC-memory-model/TMR_Exp3BSubject.txt")
    d=f.readlines()
    f.close()
    # columns: subj_num, uncued_pre,post,diff,cued_pre, cued_post, cued_diff,
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    exp3B_data=np.array(m)

    f = open("TMR_lmExp1_cued.txt")
    d= f.readlines()
    f.close()
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    data1cued=np.array(m)
#0sub_num, 1average pre_acc in strength, 2average pre_acc predicted by linear regression,
#3average post_acc, 4average pre test t in ms, 5 average post test time in ms
    data1cued[:,4]=data1cued[:,4]/60000
    data1cued[:,5]=data1cued[:,5]/60000
    data11_means=np.mean(data1cued,axis=0)

    f = open("TMR_lmExp1_uncued.txt")
    d= f.readlines()
    f.close()
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    data10=np.array(m)
#0sub_num, 1average pre_acc in strength, 2average pre_acc predicted by linear regression,
#3average post_acc, 4average pre test t in ms, 5 average post test time in ms
    data10[:,4]=data10[:,4]/60000
    data10[:,5]=data10[:,5]/60000
    data10_means=np.mean(data10,axis=0)

    f = open("TMR_lmExp2_cuedspindle.txt")
    d= f.readlines()
    f.close()
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    data22=np.array(m)
#0sub_num, 1average pre_acc in strength, 2average pre_acc predicted by linear regression,
#3average post_acc, 4average pre test t in ms, 5 average post test time in ms
    data22[:,4]=data22[:,4]/60000
    data22[:,5]=data22[:,5]/60000
    data22_means=np.mean(data22,axis=0)

    f = open("TMR_lmExp2_cuednospindle.txt")
    d= f.readlines()
    f.close()
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    data21=np.array(m)
#0sub_num, 1average pre_acc in strength, 2average pre_acc predicted by linear regression,
#3average post_acc, 4average pre test t in ms, 5 average post test time in ms
    data21[:,4]=data21[:,4]/60000
    data21[:,5]=data21[:,5]/60000
    data21_means=np.mean(data21,axis=0)

    exp3A_means=np.mean(exp3A_data,axis=0)
    exp3B_means=np.mean(exp3B_data,axis=0)

    plt.figure()
    #pass parameters in strength
    def actualPlot():
        (strength,tmr,decay) = model.estimate_tmr(data10_means[1],data10_means[3],data11_means[1],data11_means[3],
                data10_means[4],data10_means[5],data11_means[4],data11_means[5],graph=True)
        print "Data 1 Actual y in strength, decay: ",decay
        print "Data 1 TMR:", tmr
        print "Average Initial Strength Actual Data1:", strength

        tmr21=model.add_tmr(data21_means[1],data21_means[3],decay,data21_means[4],data21_means[5],label='Cued-spindle',color='yellow')
        tmr22=model.add_tmr(data22_means[1],data22_means[3],decay,data22_means[4],data22_means[5],label='Cued+spindle',color='green')
        print "TMR size data 2 Cued-spindle: ",tmr21
        print "TMR size data 2 Cued+spindle: ",tmr22

        # pre actual error bar of std
        # plt.errorbar(data10_means[4], data10_means[1], yerr=0.323715682473, fmt='o',elinewidth='2.0') #uncued pre
        # plt.errorbar(data11_means[4], data11_means[1], yerr=0.339012003076, fmt='o',elinewidth='2.0') #cued pre
        # plt.errorbar(data22_means[4], data22_means[1], yerr=0.357571016045, fmt='o',elinewidth='2.0') #cued+spindle pre actual data2
        # plt.errorbar(data21_means[4], data21_means[1], yerr=0.355922810702, fmt='o',elinewidth='2.0') #cued no spindle pre actual data2


    #predicted y
    def predictedPlot():
        (strength_lm,tmr_lm,decay_lm) = model.estimate_tmr(data10_means[2],data10_means[3],data11_means[2],data11_means[3],
                data10_means[4],data10_means[5],data11_means[4],data11_means[5],c1_label='Uncued predicted',
                        c2_label='Cued predicted',graph=True,c1_color='blue',c2_color='red',t0=data11_means[4])
        plt.plot([0,data11_means[4]],[0,data11_means[2]],color='m')
        plt.plot([0,data10_means[4]],[0,data10_means[2]],color='c')

        tmr21_lm=model.add_tmr(data21_means[2],data21_means[3],decay_lm,data21_means[4],data21_means[5],label='Cued-spindle: Predicted',color='yellow',t0=data21_means[4])
        tmr22_lm=model.add_tmr(data22_means[2],data22_means[3],decay_lm,data22_means[4],data22_means[5],label='Cued+spindle: Predicted',color='green',t0=data22_means[4])
        plt.plot([0,data21_means[4]],[0,data21_means[2]],color='yellow')
        plt.plot([0,data22_means[4]],[0,data22_means[2]],color='green')
        print "Predicted TMR size data 2 Cued-spindle: ",tmr21_lm
        print "Predicted TMR size data 2 Cued+spindle: ",tmr22_lm

        print "Data 1 Predicted y in strength, decay: ",decay_lm
        print "Data 1 TMR:", tmr_lm
        print "Average Initial Strength Predicted Data 1:", strength_lm

        # predicted pre performance error bar (std): error bar two long, DN make sense on the graph
        #plt.errorbar(data10_means[4], data10_means[2], yerr=0.293459651014, fmt='o',elinewidth='2.0') #uncued
        #plt.errorbar(data11_means[4], data11_means[2], yerr=0.301357074333, fmt='o',elinewidth='2.0') #cued predicted
        # plt.errorbar(data22_means[4], data22_means[2], yerr=0.316312930216, fmt='o',elinewidth='2.0') #cued+spindle pre predicted data2
        # plt.errorbar(data21_means[4], data21_means[2], yerr=0.325759254754, fmt='o',elinewidth='2.0') #cued no spindle pre predicted data2


    def data3Plot():
        (strength3A,tmr_size3A,decay3A)=model.estimate_tmr(model.pixels_to_str(exp3A_means[1]),model.pixels_to_str(exp3A_means[2]),
            model.pixels_to_str(exp3A_means[4]),model.pixels_to_str(exp3A_means[5]),
            c1_label='Uncued-Data3A',c2_label='Cued-Data3A',graph=True,c1_color='blueviolet',c2_color='darkgrey') #pass: pre,post(uncued),pre,post(cued)
        print "Decay data 3 Object A: ",decay3A, "TMR size: ", tmr_size3A
        print "Average Initial Strength 3A:", strength3A
        (strength3B,tmr_size3B,decay3B)=model.estimate_tmr(model.pixels_to_str(exp3B_means[1]),model.pixels_to_str(exp3B_means[2]),
            model.pixels_to_str(exp3B_means[4]),model.pixels_to_str(exp3B_means[5]),c1_label='Uncued-Data3B',c2_label='Cued-Data3B',
            graph=True,c1_color='pink',c2_color='coral') #pass: pre,post(uncued),pre,post(cued)
        print "Decay Data 3 Object B: ",decay3B, "TMR size: ", tmr_size3B
        print "Average Initial Strength 3B:", strength3B

    #actualPlot()
    #predictedPlot()
    #data3Plot()

    plt.title("Data1,2,3 with Predicted Pre-Nap Performance\nLinear Regression with Intercept = 0 Prediction")

    print "Uncued Time:",data10_means[4],data10_means[5]
    print "Cued Time:", data11_means[4],data11_means[5]

    #plt.ylim(0.35,0.7)
    plt.show()


def EEGfit(cuednum):
    subnum='19'
    #uncuednum=13
    cuednum=cuednum
    f = open("LinearRegression12/lg_"+str(subnum)+".txt")
    #f = open("LinearRegression12/TMR_lg1.txt")
    d= f.readlines()
    f.close()
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    data1cued=np.array(m)
    # 0pic_num,1m = slope,2diff=predicted - actual_y,3 percentage difference = diff/prey_lm,
    # 4prey_lm in strength,5prey in strength, 6pre_t, 7post_y in strength,
    # 8post_t, 9cued1/uncued0/spindle+cued1
    data1cued[:,6]=data1cued[:,6]/60000
    data1cued[:,8]=data1cued[:,8]/60000
    pic28=data1cued[data1cued[:,0]==cuednum][0].tolist() #cued
    #pic19=data1cued[data1cued[:,0]==uncuednum][0].tolist() #uncued

    pic19 = [1.22222222e+01,  1.55716682e-06,-5.43150874e-01,  -5.82632160e-01,   6.28309800e-01,   1.17146067e+00,9.16018657e+05,  5.12305419e-01 ,  1.47986494e+07 ,  0.00000000e+00]
    pic19[6]=pic19[6]/60000
    pic19[8]=pic19[8]/60000

    f = open("EEGtime/cueing_"+str(subnum)+".txt")
    d= f.readlines()
    f.close()
    m=[]
    for i in d:
        line=map(float,i.split())
        m.append(line)
    sub2cued=np.array(m)
    # 0subj_num,1pic_num,2sound,3tpassed sinced beginning of all practice session (assume EEG recording start immediately after pre-test),
    # 4sleep stage,5slow wave phase
    cuedpic28=sub2cued[sub2cued[:,1]==cuednum]
    tmrts=(cuedpic28[:,3]/60000).tolist()
    print pic28

    #predicted pre, multi tmr
    plt.figure()
    (strength,tmr,decay) = model.estimate_multi_tmr(pic19[4],pic19[7],pic28[4],pic28[7],
                    pic19[6],pic19[8], pic28[6],pic28[8],tmrts,graph=True,c1_label='Uncued Multiple TMR',
                    c2_label='Cued Multiple TMR',c2_color='green')
    #print "tmr timings",tmrts,"pre,post acc",pic28[4],pic28[7],"pre, post time",pic28[6],pic28[8],
    print "Uncued Predicted y in strength, decay: ",decay
    print "TMR multiple events:", tmr
    print "Number of cueing",len(tmrts)

    #predicted pre, 1tmr
    (strength,tmr,decay) = model.estimate_multi_tmr(pic19[4],pic19[7],pic28[4],pic28[7],
                    pic19[6],pic19[8], pic28[6],pic28[8],[pic28[6]+45.0],graph=True,c1_label='Uncued 1TMR',c2_label='Cued 1TMR',
                    c1_color='blue',c2_color='m')
    plt.plot([0,pic19[6]],[0,pic19[4]],color='blue')
    plt.plot([0,pic28[6]],[0,pic28[4]],color='m')
    plt.xlim(-10,300)
    plt.ylim(-0.1,1.3)

    print "1 TMR:", tmr,"\n"
   # print "Average Initial Strength Predicted:", strength


    # plt.title("1 Cueing Event")
    plt.title("Multiple Cueing Events vs 1 cue\n 1 cue assumed at 45 min in nap")

    def scatterPlot(subnum,pic_cued,pic_uncued):
        f = open("LinearRegression12/"+str(subnum)+"y.txt")
        d= f.readlines()
        f.close()
        m=[]
        for i in d:
            line=map(float,i.split())
            m.append(line)
        scattery=m
        f = open("LinearRegression12/"+str(subnum)+"x.txt")
        d= f.readlines()
        f.close()
        m=[]
        for i in d:
            line=map(float,i.split())
            temp=[]
            for l in line:
                temp.append(l/60000)
            m.append(temp)
        scatterx=m
        plt.scatter(scatterx[pic_cued-1],scattery[pic_cued-1],color='m')
        plt.scatter(scatterx[pic_uncued-1],scattery[pic_uncued-1],color='blue')

    #plt.show()
    # plt.xlim(15,59)
    # plt.ylim(0.8,1.1)
    scatterPlot(subnum,cuednum) #uncuednum
    #plt.show()



#fitmodel()
#writeAll()
# for i in [1,3,6,9,10,13,15,16,17,20,22,23,26]:
#     EEGfit(i)
EEGfit(3)
plt.show()

def groupVector():
    arrayshape = np.arange(-5,300)
    cuedAlly=arrayshape*1
    cuedAllt=cuedAlly*1
    uncuedt = arrayshape*1
    uncuedy = arrayshape*1

    #fl=glob.glob("EEGtime/cueing_[0-9][0-9].txt")
    fl=glob.glob("EEGtime/cueing_[0-9][0-9][0-9].txt")
    for i in fl:
        print i
        subnum=i[-7:-4]
        f= open(i)
        d= f.readlines()
        f.close()
        m=[]
        for i in d:
            line=map(float,i.split())
            m.append(line)
        subcued=np.array(m)
        # 0subj_num,1pic_num,2sound,3tpassed sinced beginning of all practice session (assume EEG recording start immediately after pre-test),
        # 4sleep stage,5slow wave phase

        f = open("LinearRegression12/lg_"+str(subnum)+".txt")
        #f = open("LinearRegression12/lg_19.txt")
        d= f.readlines()
        f.close()
        m=[]
        for i in d:
            line=map(float,i.split())
            m.append(line)
        data1cued=np.array(m)
        data1cued[:,6]=data1cued[:,6]/60000
        data1cued[:,8]=data1cued[:,8]/60000
        raw_uncued=data1cued[data1cued[:,9]==0]
        data1cued=data1cued[data1cued[:,9]==2]

        uncued = [1.22222222e+01,  1.55716682e-06,-5.43150874e-01,  -5.82632160e-01,   6.28309800e-01,   1.17146067e+00,9.16018657e+05,  5.12305419e-01 ,  1.47986494e+07 ,  0.00000000e+00]
        uncued[6]=uncued[6]/60000
        uncued[8]=uncued[8]/60000

    #0pic_num,1m = slope,2diff=predicted - actual_y,3 percentage difference = diff/prey_lm,
    #4prey_lm in strength,5prey in strength, 6pre_t, 7post_y in strength,
    #8post_t, 9cued1/uncued0/spindle+cued1

        decay = -0.000882163008012 #float(decay)/np.shape(uncued)[0]
        subcuedy=arrayshape*1
        subcuedt=arrayshape*1

        for p in data1cued:
            print p
            C=model.MemoryTrace(p[4],p[6])
            C.forgetting=decay
            currpic=subcued[subcued[:,1]==p[0]]
            tmrts=(currpic[:,3]/60000).tolist()
            #print p[0],tmrts
            C.find_multi_tmr2(tmrts,p[8],p[7])
            t,y=C.traceVector(step=1)
            #print p,t,y
            subcuedt = np.vstack([subcuedt,t])
            subcuedy = np.vstack([subcuedy,y])
        subcuedt = np.delete(subcuedt,0,0)
        subcuedy = np.delete(subcuedy,0,0)
        subtvector=np.mean(subcuedt,axis=0)
        subyvector=np.mean(subcuedy,axis=0)
        cuedAlly = np.vstack([cuedAlly,subyvector])
        cuedAllt = np.vstack([cuedAllt,subtvector])

    plt.figure()
    C=model.MemoryTrace(uncued[4],uncued[6])
    C.find_decay(uncued[7],uncued[8])
    C.graph([-5,300],legend='uncued')

    cuedAllt = np.mean(np.delete(cuedAllt,0,0),axis=0)
    cuedAlly = np.mean(np.delete(cuedAlly,0,0),axis=0)

    # uncuedAllt = np.mean(np.delete(uncuedt,0,0),axis=0)
    # uncuedAlly = np.mean(np.delete(uncuedy,0,0),axis=0)
    # plt.plot(uncuedAllt,uncuedAlly,color='green',label='Uncued Vector Result')

    plt.plot(cuedAllt,cuedAlly,color='red',label='cued+spindle')
    plt.legend(loc='best')
    plt.show()


#groupVector()