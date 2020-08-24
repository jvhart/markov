
import numpy as np
from scipy import sparse

def comm_classes(P):
    '''
    Parameters:
        P - scipy.sparse.csr_matrix
            Markov chain transition matrix, rows sum to 1

    Outputs:
        classes - list
            List of lists describing the communication classes of the Markov chain
        types - list
            List of communication class types, Transient or Absorbing, corresponding to classes output
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
    return classes, types

def decompose(P,classes,types):
    '''
    Parameters:
        P - scipy.sparse.csr_matrix
            Markov chain transition matrix, rows sum to 1
        classes - list
            List of lists describing the communication classes of the Markov chain
        types - list
            List of communication class types, Transient or Absorbing, corresponding to classes output

    Outputs:
        diagonal - list
            list of scipy.sparse.csr_matrix objects along the diagonal for sub-markov chains
        upper_triangle - list
            list of scipy.sparse.csr_matrix objects upper triangle matrices indicating transient class escape probabilities
    '''
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
    Inputs:
        diagonal - list
            list of scipy.sparse.csr_matrix objects along the diagonal for sub-markov chains
        upper_triangle - list
            list of scipy.sparse.csr_matrix objects upper triangle matrices indicating transient class escape probabilities

    Outputs:
        A - scipy.sparse.csr_matrix
            Markov chain transition matrix in canonocal format
        Q - scipy.sparse.csr_matrix
            Transient portionof the Markov chain
        R - scipy.sparse.csr_matrix
            Escape probability matrix from the transient portion to absorbing
        I - scipy.sparse.csr_matrix
            Absorbing class submatrix, itself a transition matrix

        A = (Q | R)
            (0 | I)
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
