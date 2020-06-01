'''
Author: Jarod Hart
Date: 9/28/2018

Description:  This is a package for simulating and analyzing discrete-time,
finite state space, Markov chains.  The theory supporting these calculations
can be found in just about any textbook on Markov chains, stochastic processes,
or probability theory, among many other places.  For a quick reference, you
can take a look at the chapter from Grinstead and Snell's textbook Introduction
to Probability on Markov chains, linked below.  The notation used in this
package was taken from this resource.

https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf

Example: Analysis of a simple random walk.

from markov import MarkovChain
p = .5
num_states = 20
P = np.zeros((num_states,num_states))
P[0,0] = 1
P[-1,-1] = 1
for i in range(1,num_states-1):
    P[i,i-1] = .5
    P[i,i+1] = .5
RW = MarkovChain(P=P)
RW.analyze()

RW.N[5,2]   # Expected number of times to visit state 2 starting from state 5
RW.B[6,1]   # Probability of being absorbed into state 1 starting from state 6
RW.T[3,0]   # Expected time to absorption, starting from state 3

x = RW.simulate(n=200,initial=8) # Simulate the Markov chain for 200 time steps
                                 # with initial state 8.

'''

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv as sp_inv

def comm_classes(P):
    '''
    Takes as input a square matrix, defined as a dense matrix, csr_matrix, or
    other matrix formats supported by the sparse matrix packages in scipy, and
    returns the communication classes of the associated Markov chain.  The
    classes are defined in a list of lists made up of the indices that indicate
    the class members.  It also returns the types of classes, either transient
    or absorbing, in a list of strings.  The classes list and types lists
    returned are the same size, and corresponding elements indicate the class
    type for each communication class.
    '''
    if not isinstance(P,sparse.csr_matrix):
        P = sparse.csr_matrix(P)
    rows,cols = P.nonzero()

    num_classes,class_map = sparse.csgraph.connected_components(P,directed = True,connection = 'strong')

    CC = []
    for i in range(num_classes):
        CC += [list(np.where(class_map == i)[0])]

    Ts = []
    transient_classes = np.array([])
    for i in range(num_classes):
        if sum([P[CC[i][0],j] for j in CC[i]]) < 1:
            Ts += ['Transient']
            transient_classes = np.append(transient_classes,i)
        else:
            Ts += ['Absorbing']
    transient_classes = [int(i) for i in transient_classes]
    order = list(transient_classes) + [int(i) for i in range(num_classes) if not i in transient_classes]
    classes = []
    types = []
    count = 0
    for i in order:
        count += 1
        classes += [CC[i]]
        types += [Ts[i]]
    return classes,types

def decompose(P):
    '''
    Takes as input a square matrix, defined as a dense matrix, csr_matrix, or
    other matrix formats supported by the sparse matrix packages in scipy, and
    returns the diagonal block matrices and upper triangle matrices
    representing the transition probabilities within and leaving all transient
    classes in the chain.  The diagonal list also contain matrices representing
    the transition probabilities within absorbing classes.  The classes and
    class types are also returned, which are exactly the ones computed in the
    comm_class function.
    '''
    classes,types = comm_classes(P)
    num_classes = len(classes)
    state_order = []
    for i in range(num_classes):
        state_order += classes[i]

    diagonal = []
    upper_triangle = []

    rows,cols = P.nonzero()
    for i in range(num_classes):
        diagonal += [P[classes[i],:][:,classes[i]]]
        if types[i] == 'Transient':
            state_order = [j for j in state_order if not j in classes[i]]
            upper_triangle += [P[classes[i],:][:,state_order]]
    diagonal = [sparse.csr_matrix(d) for d in diagonal]
    upper_triangle = [sparse.csr_matrix(d) for d in upper_triangle]
    return diagonal,upper_triangle,classes,types

def construct_canonical_matrix(diagonal,upper_triangle):
    '''
    Takes two lists of matrices diagonal and upper_triangle associated to the
    decomposition of a Markov chain.  Uses this information to construct the
    transition matrix for the same Markov chain with state label permuted in
    a canonical way for the chain.  It lists all transient states first
    organized by communication class, and then all absorbing states organized
    by communication class.  This canonical transition matrix is returned as
    the csr_matrix A.  Q, R, and I are the standard components of the cononcial
    form transition matrix.  Q represents the tranistion probaiblitilties
    within all transient classes, R represents the probabilities of leaving
    transient classes to absorbing classes, and I represents the transition
    probabilities within the absorbing classes.  These objects can be easily
    used to compute absorption probabilities, expected absorption times,
    first passage times, return times, etc.
    '''
    N = sum([d.shape[0] for d in diagonal])
    A = np.zeros((N,N))
    Tstates = 0
    current_loc = 0
    for i in range(len(diagonal)):
        if i < len(upper_triangle):
            m,n = upper_triangle[i].shape
            A[current_loc:current_loc+m,current_loc+m:current_loc+m+n] = upper_triangle[i].todense()
            Tstates += m
        else:
            m = diagonal[i].shape[0]
        A[current_loc:current_loc+m,current_loc:current_loc+m] = diagonal[i].todense()
        current_loc += m
    Q = sparse.csr_matrix(A[:Tstates,:Tstates])
    R = sparse.csr_matrix(A[:Tstates,Tstates:])
    I = sparse.csr_matrix(A[Tstates:,Tstates:])
    A = sparse.csr_matrix(A)
    return A,Q,R,I

class MarkovChain():
    '''
    This is a Markov chain object.  It will take as inputs the transition
    matrix and initial distribution to define a Markov chain.
    '''
    def __init__(self,**kwargs):
        if 'P' in list(kwargs.keys()):
            self.P = kwargs['P']
        else:
            self.P = np.zeros((20,20))
            self.P[0,0] = 1
            self.P[19,19] = 1
            for i in range(1,19):
                self.P[i,i-1] = .5
                self.P[i,i+1] = .5
        if not isinstance(self.P,sparse.csr_matrix):
            self.P = sparse.csr_matrix(sparse.csr_matrix(self.P))
        self.num_states = self.P.shape[0]

        if 'initial' in list(kwargs.keys()):
            self.initial = kwargs['initial']
        else:
            self.initial = np.ones(self.P.shape[0])/(self.P.shape[0])

    def analyze(self,**kwargs):
        '''
        This method does the heavy-lifting to analyze the Markov chain.  It
        calls the functions defined above to compute the canonical form of the
        transition matrix, and uses this to compute the fundamental matrix,
        absorption probability matrix, and expected time to absorption matrix.
        It also calculates *the stationary* distribution.

        *If the chain has a unique absorbing state, then the chain has a unique
        stationary vector.  Otherwise, a stationary vector is computed for each
        absorbing state, concatinated, and normalized to create a stationary
        vector that has nonzero entries corresponding to every absorbing
        state.*
        '''
        if 'P' in list(kwargs.keys()):
            self.P = kwargs['P']

        if 'initial' in list(kwargs.keys()):
            self.initial = kwargs['initial']
        elif len(self.initial) != self.P.shape[0]:
            self.initial = np.ones(self.P.shape[0])/(self.P.shape[0])
        if not isinstance(self.P,sparse.csr_matrix):
            self.P = sparse.csr_matrix(sparse.csr_matrix(self.P))

        self.num_states = self.P.shape[0]

        self.diagonal,self.upper_triangle,self.classes,self.types = decompose(self.P)
        self.canonical_P,self.Q,self.R,self.I = construct_canonical_matrix(self.diagonal,self.upper_triangle)

        self.canonical_order = []
        for c in self.classes:
            self.canonical_order += c

        self.map = {i : np.where(self.canonical_order == i)[0] for i in range(self.num_states)}

        try:
            self.N = sp_inv( sparse.identity(self.Q.shape[0]).tocsc() - self.Q.tocsc()).tocsr()
        except:
            self.N = sparse.csr_matrix(np.linalg.inv( (sparse.identity(self.Q.shape[0]).tocsc() - self.Q.tocsc()).todense()))
        self.B = self.N*self.R
        self.T = self.N*sparse.csc_matrix(np.ones((self.Q.shape[0],1)))
        self.stationary = np.array((1,0))

        if len(self.classes) == 1:
            self.types = ['Absorbing']
            w,v = np.linalg.eig(self.P.todense().T)
            index_order = np.argsort(np.abs(w))[::-1]
            w = w[index_order]
            v = v[:,index_order]
            self.stationary = v[:,0]
        else:
            self.stationary = np.zeros(self.num_states)
            for i in range(len(self.classes)):
                if self.types[i] == 'Absorbing':
                    w,v = np.linalg.eig(np.array([[self.P.todense()[j,k] for k in self.classes[i]] for j in self.classes[i]]).T)
                    index_order = np.argsort(np.abs(w))[::-1]
                    w = w[index_order]
                    v = v[:,index_order]
                    self.stationary[self.classes[i]] = v[:,0]
        self.stationary = self.stationary / sum(self.stationary)
        return self

    def simulate(self,**kwargs):
        '''
        Takes as input the number of time steps to simulate and an initial
        state or initial distribution.  It returns an integer numpy array with
        the states visited.
        '''
#        if 'mode' in list(kwargs.keys()):
#            mode = kwargs['mode']
#        else:
#            mode = iteration_limit
        if 'initial' in list(kwargs.keys()):
            self.initial = kwargs['initial']
            if isinstance(self.initial,int):
                temp = self.initial
                self.initial = np.zeros(self.P.shape[0])
                self.initial[temp] = 1
            self.initial = self.initial/sum(self.initial)

        if not isinstance(self.P,sparse.csr_matrix):
            self.P = sparse.csr_matrix(sparse.csr_matrix(self.P))

        if 'n' in list(kwargs.keys()):
            n = kwargs['n']
        else:
            n = 100
        F = np.array([np.cumsum(self.P.getrow(i).data) for i in range(self.P.shape[0])])
        R = np.random.rand(n-1)
        x = np.zeros(n,dtype = int)
        x[0] = np.random.choice(a=range(self.num_states),size=1,p=self.initial)[0]
        for i,r in zip(range(1,n),R):
            x[i] = self.P.getrow(x[i-1]).nonzero()[1][sum(1*(F[x[i-1]]<r))]
        return x
