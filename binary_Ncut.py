import numpy as np
from sklearn.metrics import accuracy_score
try:
    import _pickle as pickle
except:
    import pickle

# helper function
def load_data():

    with open("pos_covote_2.pkl", "rb") as fin:
        pos_covote = pickle.load(fin)
    with open("neg_covote_2.pkl", "rb") as fin:
        neg_covote = pickle.load(fin)
    f = open('Dict_Person_Senate','r')
    parties_set = set()
    idtoperson = {}
    persontoid = {}
    idtoparty = {}
    for line in f:
        line1 = line.split('\t')
        [id,firstname,lastname,party,_] = line1
        if party == 'ID':
            party = 'Independent'
        idtoperson[int(id)] = firstname + ' ' + lastname
        persontoid[firstname + ' ' + lastname] = int(id)
        idtoparty[int(id)] = party
        if party not in parties_set:
            parties_set.add(party)
    
    f = open('Dict_Person_House','r')
    for line in f:
        line1 = line.split('\t')
        [id,first,party,_,_] = line1
        if party == 'ID':
            party = 'Independent'
        idtoparty[int(id)] = party
        idtoperson[int(id)] = first
        persontoid[first] = int(id)
        if party not in parties_set:
            parties_set.add(party)
    return pos_covote, neg_covote, idtoperson, persontoid, idtoparty

def main():
    pos_covote, neg_covote, idtoperson, persontoid, idtoparty = load_data()
    n = len(idtoperson)
    W = np.zeros((n,n))

    for vote in pos_covote:
        (i,j) = vote
        W[i,j] += pos_covote[vote]
        W[j,i] += pos_covote[vote]

    for vote in neg_covote:
        (i,j) = vote
        W[i,j] += neg_covote[vote]
        W[j,i] += neg_covote[vote]

    D = np.zeros((n,n))
    for i in range(0,n):
        D[i,i] += np.sum(W[:,i])

    L = D - W
    temp = np.matmul(np.linalg.inv(D**0.5),L)
    L_rw = np.matmul(temp,np.linalg.inv(D**0.5))

    (eval,evec)=np.linalg.eig(L_rw)
    e_sorted = np.sort(eval)
    idx = np.where(eval == e_sorted[1])[0][0]
    q = evec[idx]
    plist = []
    tlist = []
    for id in range(0,n):
        if idtoparty[id] == 'Independent':
            continue 
        if q[id] > 0:
            pre = 1
        else:
            pre = 0 
        if idtoparty[id] == 'R':
            tru = 1
        else:
            tru = 0 
        plist.append(pre)
        tlist.append(tru)
    ypred = np.array(plist)
    ytrue = np.array(tlist)

    accur = accuracy_score(ytrue,ypred)
    print('accuracy score is:')
    print accur
