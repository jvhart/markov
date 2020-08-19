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
    for i in order:
        classes += [CC[i]]
        types += [Ts[i]]
    return classes,types

def decompose(P,classes,types):
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
    # classes,types = comm_classes(P)
    num_classes = len(classes)
    state_order = []
    for i in range(num_classes):
        state_order += classes[i]

    diagonal = []
    upper_triangle = []
    for i in range(num_classes):
        diagonal += [P[classes[i],:][:,classes[i]]]
        if types[i] == 'Transient':
            state_order = [j for j in state_order if not j in classes[i]]
            upper_triangle += [P[classes[i],:][:,state_order]]
    diagonal = [sparse.csr_matrix(d) for d in diagonal]
    upper_triangle = [sparse.csr_matrix(d) for d in upper_triangle]
    return diagonal,upper_triangle

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

class MarkovChainState:
    next_state_id = 0

    def __init__(self,**kwargs):
        self.state_id = self.next_state_id
        self.next_state_id += 1

        for k,v in kwargs.items():
            setattr(self,k,v)


class MarkovChainEdge:
    next_edge_id = 0

    def __init__(self,orig_state,term_state,transition_prob,**kwargs):
        self.edge_id = self.next_edge_id
        self.next_edge_id += 1

        self.orig_state = orig_state
        self.term_state = term_state
        self.transition_prob = transition_prob
        for k,v in kwargs.items():
            setattr(self,k,v)


class MarkovChain:
    '''
    This is a Markov chain object.  It will take as inputs the transition
    matrix and initial distribution to define a Markov chain.
    '''
    def __init__(self,**kwargs):
        self.state_ordinal = 0
        self.states = {}
        self.edges = []
        for k,v in kwargs.items():
            setattr(self,k,v)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def transition_matrix_is_valid(self):
        valid = True
        for st in self.states.keys():
            valid = round(sum([edge.transition_prob for edge in self.edges if edge.orig_state.state_ordinal == st]),8) == 1
            if not valid:
                break
        return valid

    @property
    def rows(self):
        return [edge.orig_state.state_ordinal for edge in self.edges]

    @property
    def cols(self):
        return [edge.term_state.state_ordinal for edge in self.edges]

    @property
    def vals(self):
        return [edge.transition_prob for edge in self.edges]

    @property
    def is_ergodic(self):
        if not hasattr(self,'classes'):
            self.get_communication_classes()
        if len(self.classes) == 1:
            return True
        return False

    def from_rcv(self,rows,cols,vals):
        num_states = 1 + max(list(rows) + list(cols))
        for i in range(num_states):
            self.add_state()
        for r,c,v in zip(rows,cols,vals):
            self.add_edge(orig_state=self.states.get(r),term_state=self.states.get(c),transition_prob=v)

    def from_nparray(self,A):
        rows,cols = np.where(A > 0)
        vals = [A[r,c] for r,c in zip(rows,cols)]
        self.from_rcv(rows,cols,vals)

    def add_state(self,**kwargs):
        state = MarkovChainState(**kwargs)
        state.state_ordinal = self.state_ordinal
        self.states.update({self.state_ordinal : state})
        self.state_ordinal += 1
        return state

    def add_edge(self,**kwargs):
        edge = MarkovChainEdge(**kwargs)
        self.edges += [edge]

    def get_transition_matrix(self,format='csr'):
        if format == 'csr':
            n = self.num_states
            self.csr_transition_matrix = sparse.csr_matrix( (self.vals,(self.rows,self.cols)), shape=(n,n) )
            return self.csr_transition_matrix
        if format == 'list':
            self.list_transition_matrix = [(r,c,v) for r,c,v in zip(self.rows,self.cols,self.vals)]
            return self.list_transition_matrix
        if format == 'np':
            n = self.num_states
            P = np.zeros((n,n))
            for r,c,v in zip(self.rows,self.cols,self.vals):
                P[r,c] = v
            self.np_transition_matrix = P
            return P

    def get_communication_classes(self):
        if not hasattr(self,'csr_transition_matrix'):
            self.get_transition_matrix()
        if getattr(self,'csr_transition_matrix') is None:
            self.get_transition_matrix()
        self.classes, self.types = comm_classes(self.csr_transition_matrix)
        self.canonical_state_map = {}
        j = 0
        for ls in self.classes:
            for i in ls:
                self.canonical_state_map.update({i : j})
                j += 1
        return self.classes, self.types

    def get_transition_matrix_decomposition(self):
        if not hasattr(self,'csr_transition_matrix'):
            self.get_transition_matrix()
        if not (hasattr(self,'classes') and hasattr(self,'types')):
            self.get_communication_classes()
        self.diagonal, self.upper_triangle = decompose(self.csr_transition_matrix,self.classes, self.types)
        return self.diagonal, self.upper_triangle

    def get_canonical_transition_matrix(self):
        if not (hasattr(self,'diagonal') and hasattr(self,'upper_triangle')):
            self.get_transition_matrix_decomposition()
        self.canonical_transition_matrix, self.Q, self.R, self.I = construct_canonical_matrix(self.diagonal,self.upper_triangle)
        return self.canonical_transition_matrix, self.Q, self.R, self.I

    def get_expected_visits_matrix(self):
        if not hasattr(self,'Q'):
            self.get_canonical_transition_matrix()
        try:
            self.N = sp_inv( sparse.identity(self.Q.shape[0]).tocsc() - self.Q.tocsc()).tocsr()
        except:
            self.N = sparse.csr_matrix(np.linalg.inv( (sparse.identity(self.Q.shape[0]).tocsc() - self.Q.tocsc()).todense()))
        return self.N

    def get_absorption_probability_matrix(self):
        if not hasattr(self,'R'):
            self.get_canonical_transition_matrix()
        if not hasattr(self,'N'):
            self.get_expected_visits_matrix()
        self.B = self.N * self.R
        return self.B

    def get_expected_time_to_absorption_matrix(self):
        if not hasattr(self,'N'):
            self.get_expected_visits_matrix()
        if not hasattr(self,'Q'):
            self.get_canonical_transition_matrix()
        self.T = self.N * sparse.csc_matrix(np.ones((self.Q.shape[0],1)))
        return self.T

    def get_stationary_vector(self):
        self.get_communication_classes()
        if self.is_ergodic:
            w,v = np.linalg.eig(self.csr_transition_matrix.todense().T)
            index_order = np.argsort(np.abs(w))[::-1]
            w = w[index_order]
            v = v[:,index_order]
            self.stationary_vector = np.array(v[:,0])[:,0]
            return self.stationary_vector
        else:
            num_abs_classes = sum([1 * (tp == 'Absorbing') for tp in self.types])
            self.stationary_vector = np.zeros(self.num_states)
            for cls,tp in zip(self.classes,self.types):
                if tp == 'Transient':
                    continue
                sub_mc = MarkovChain()
                sub_mc_map = {}
                for st in cls:
                    state_obj = sub_mc.add_state(parent_state_ordinal=st)
                    sub_mc_map.update({st : state_obj})
                for r,c,v in zip(self.rows,self.cols,self.vals):
                    if r in cls and c in cls:
                        sub_mc.add_edge(orig_state=sub_mc_map.get(r),term_state=sub_mc_map.get(c),transition_prob=v)
                sub_stationary_vector = sub_mc.get_stationary_vector()
                sub_stationary_vector = sub_stationary_vector / np.sum(sub_stationary_vector)
                for k,v in sub_mc_map.items():
                    self.stationary_vector[k] = sub_stationary_vector[v.state_ordinal] / num_abs_classes

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
        if 'initial_state' in kwargs.keys():
            self.initial = kwargs.get('initial_state')
            if isinstance(self.initial_state,int):
                temp = self.initial_state
                self.initial_state = np.zeros(self.get_transition_matrix(format='csr').shape[0])
                self.initial_state[temp] = 1
            self.initial_state = self.initial_state/sum(self.initial_state)
        elif not hasattr(self,'initial_state'):
            self.initial_state = np.ones(self.num_states) / self.num_states

        if not hasattr(self,'csr_transition_matrix'):
            self.get_transition_matrix()
        if 'n' in list(kwargs.keys()):
            n = kwargs['n']
        else:
            n = 100
        F = np.array([np.cumsum(self.csr_transition_matrix.getrow(i).data) for i in range(self.csr_transition_matrix.shape[0])])
        R = np.random.rand(n-1)
        x = np.zeros(n,dtype = int)
        x[0] = np.random.choice(a=range(self.num_states),size=1,p=self.initial_state)[0]
        for i,r in zip(range(1,n),R):
            x[i] = self.csr_transition_matrix.getrow(x[i-1]).nonzero()[1][sum(1*(F[x[i-1]]<r))]
        return x
