def stdyst_cxt(arr):
    Mqprod_stt = np.zeros(T.shape[0] - t_0, len(qmode_arr))
    for kn, k in enumerate(qmode_arr):
        Mq_, Mq_3 = calc_mkt(k, steps)
        #Mq_stt  = Mq_[t_0:]
        #Mqprod_stt = np.zeros(Mq_stt.shape)
        
        for tn, t in enumerate(range(0, T[-1]-t_0)):
            Mqprod_stt[tn,kn] = np.sum(Mq_stt*np.roll(Mq_stt,-tn,axis=0))/(T[-1] - (t_0 -t))
        
    return Mqprod_stt
            
            
