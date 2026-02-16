### Basis class, 

import numpy as np
from hydrogen_sim.orbitals import Orbital

def _make_hydrogen_orbitals(nmax):

    orbs_list = []

    for n in range(1,nmax+1):
        for l in range(n):
            if 0 <= l <= 3:
                for m in range(-l,l+1):
                    orbs_list.append(Orbital(n,l,m))
            
            else:
                continue

    return orbs_list

class Basis:
    def __init__(self, orbitals):
        self.orbitals = list(orbitals)
        self.N = len(self.orbitals)

        self.key_to_index = {
            orb.key(): i for i, orb in enumerate(self.orbitals)
        }

    # accessing
    def numbers_to_index(self, n = None, l = None, m = None, key = None):
        if key is not None:
            return self.key_to_index[key]
        return self.key_to_index[(n,l,m)]
    
    def index_to_numbers(self,i):
        return self.orbitals[i].key
    
    # selecting orbitals
    def select(self, predicate):
        """
        Return indices i for which predicate(orbital) is True
        """
        return [i for i, orb in enumerate(self.orbitals) if predicate(orb)]

    def select_n(self, n):
        return self.select(lambda o: o.n == n)

    def select_l(self, l):
        return self.select(lambda o: o.l == l)

    def select_nl(self, n, l):
        return self.select(lambda o: o.n == n and o.l == l)

def make_hydrogen_basis(nmax):
    return Basis(_make_hydrogen_orbitals(nmax))