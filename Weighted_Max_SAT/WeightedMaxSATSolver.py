import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.primitives import Sampler
from qiskit.compiler import transpile, assemble
from math import floor, log, sqrt, pi, ceil
from IPython.display import display
from random import randint
import sys
import time

from sat_circuit import SATCircuit
from cnf_reader import CNFReader

class WeightedMaxSATCircuit:

    def __init__(self, cnf, it, current_minimum):

        # Format of the circuit : qubit for the variables - integer register - sign qubit - ancilla qubit

        self.cnf = cnf
        self.__int_size = 1 + floor(log(sum(cnf.get_weights()), 2))
        self.qr = QuantumRegister(cnf.get_num_vars() + self.__int_size + 2, 'q')
        self.cr = ClassicalRegister(self.__int_size + 1, 'c')
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self.SAT = SATCircuit(cnf)
        self.it = it
        self.current_minimum = current_minimum

        # Blocks to apply the algorithm

        self.A = self.get_A()
        self.O = self.get_O()
        self.D = self.get_D()

        self.build_circuit()

    def get_A(self): # Algorithm
        # This method's purpose is to get the block corresponding to the Weighted Max-SAT algorithm
        A = QuantumCircuit(self.cnf.get_num_vars() + self.__int_size + 2)

        max_sum = sum(self.cnf.get_weights())
        for i in range(self.__int_size):
            if max_sum // 2**i % 2 == 1:
                A.x(self.cnf.get_num_vars() + i)

        A.append(self.SAT.to_gate(), range(cnf.get_num_vars() + self.__int_size + 2))

        # Take out the current minimum

        for j in range(self.__int_size + 1):
            if (self.current_minimum // 2 ** j) % 2 == 1:
                for k in range(self.__int_size + 1 - j):
                    lam = -np.pi / (2 ** k)
                    A.p(lam, self.cnf.get_num_vars() + j + k) # TODO verif

        transformer = QFT(self.__int_size + 1, do_swaps=False).to_gate().inverse()
        A.append(transformer, range(self.cnf.get_num_vars(), self.cnf.get_num_vars() + self.__int_size + 1))

        return A

    def get_O(self): # Oracle
        O = QuantumCircuit(self.cnf.get_num_vars() + self.__int_size + 2)
        O.cz(-2, -1)

        return O

    def get_D(self): # Diffusion operator
        D = QuantumCircuit(self.cnf.get_num_vars() + self.__int_size + 2)
        D.x(range(self.cnf.get_num_vars() + self.__int_size + 2))
        D.h(-1)
        D.mcx([i for i in range(cnf.get_num_vars() + self.__int_size + 1)], -1)
        D.h(-1)
        D.x(range(self.cnf.get_num_vars() + self.__int_size + 2))

        return D
    
    def build_circuit(self):

        self.circuit.append(self.A.to_gate(label = "$A$"), range(self.cnf.get_num_vars() + self.__int_size + 2))

        for i in range(max(self.it,1)):
            self.circuit.append(self.O.to_gate(label = "$O$"), range(self.cnf.get_num_vars() + self.__int_size + 2))
            self.circuit.append(self.A.inverse().to_gate(label = "$A*$"), range(self.cnf.get_num_vars() + self.__int_size + 2))
            self.circuit.append(self.D.to_gate(label = "$D$"), range(self.cnf.get_num_vars() + self.__int_size + 2))
            self.circuit.append(self.A.to_gate(label = "$A$"), range(self.cnf.get_num_vars() + self.__int_size + 2))

        # Measure

        self.circuit.measure(range(self.cnf.get_num_vars(), self.cnf.get_num_vars() + self.__int_size + 1), self.cr)

        transpile(self.circuit, optimization_level=2)

    def draw(self, decomposition_level=0, show_info=False):
        c = self.circuit.copy()
        while decomposition_level > 0:
            c = c.decompose()
            decomposition_level -= 1
        if show_info:
            print("depth := " + str(c.depth()))
            print("size := " + str(c.size()))
        return c.draw()

class WeightedMaxSATSolver:

    def __init__(self, cnf, stop=4):
        self.cnf = cnf
        self.current_minimum = self.get_initial_minimum()
        self.stop = stop
        self.not_better = 0

        self.statistics = dict()
        self.it = 0

        self.current_depth = 0

    def get_initial_minimum(self):
        # Get the score if the all the variables are set to true (arbitrary)
        score = sum(self.cnf.get_weights())
        for i in range(self.cnf.get_num_clauses()):
            clause = self.cnf.get_clauses()[i]
            weight = self.cnf.get_weights()[i]
            for v in clause:
                if v > 0:
                    score -= weight
                    break
        return score

    def solve(self):

        start = time.time()

        k = 1
        kmax = floor((4 * sqrt(2**self.cnf.get_num_vars()) / pi))
        
        self.statistics[(self.it, "score")] = sum(self.cnf.get_weights()) - self.current_minimum
        print(str(self.cnf.get_num_vars())+','+str(self.cnf.get_num_clauses())+','+str(sum(self.cnf.get_weights()) - self.current_minimum)+','+str(time.time() - start)+','+str(0))
        self.it += 1

        while self.not_better < self.stop:
            nit = randint(1, min(k, kmax))
            it_circuit = WeightedMaxSATCircuit(cnf, nit, self.current_minimum)
            #self.current_depth += it_circuit.circuit.copy().decompose().decompose().decompose().decompose().decompose().decompose().depth() # VERY SLOW, uncomment only if needed for analysis
            sampler = Sampler()
            result = list(sampler.run(it_circuit.circuit, skip_transpilation=True, shots = 1).result().quasi_dists[0])
            result = result[0]
            if result < self.current_minimum:
                self.not_better = 0
                self.current_minimum = result
            else:
                self.not_better += 1
                k = ceil(1.25 * kmax)
            self.statistics[(self.it, "score")] = sum(self.cnf.get_weights()) - self.current_minimum
            print(str(self.cnf.get_num_vars())+','+str(self.cnf.get_num_clauses())+','+str(sum(self.cnf.get_weights()) - self.current_minimum)+','+str(time.time() - start)+','+str(self.current_depth))
            self.it += 1

if __name__ == "__main__":
    try:
        cnf = CNFReader(sys.argv[1])
    except:
        print("Usage for custom instances : python3 WeightedMaxSATSolver.py <cnf_file>")
        print("Default instance instead")
        cnf = CNFReader("../data/test5.cnf")
        sys.exit(1)
    solver = WeightedMaxSATSolver(cnf)
    solver.solve()
