from cnf_reader import CNFReader
import sys
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Sampler
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from IPython.display import display  # Display in command line
from math import floor, log
from collections import defaultdict
import matplotlib.pyplot as plt

def defaultzero():
    return 0

class SATCircuit:

    def __init__(self, cnf):
        self.cnf = cnf
        self.__int_size = 1 + floor(log(sum(cnf.get_weights()), 2))
        self.qr = QuantumRegister(cnf.get_num_vars() + self.__int_size + 2, 'q')
        self.circuit = QuantumCircuit(self.qr)
        self.build_circuit()

    """
    Build a quantum adder block to add the given number to a fixed size register in QFT form
    Most significant bit is at the end of the register
    """
    def adder_block(self, number, flags):
        name = "Adder_" + str(number)
        adder = QuantumCircuit(self.cnf.get_num_vars() + self.__int_size + 2, name=name)

        controls = []
        for i in range(len(flags)):
            if flags[i] != 0:
                controls.append(i)

        for i in range(len(flags)):
            if flags[i] == 1:
                adder.x(i)

        phase_diff = defaultdict(defaultzero)

        for j in range(self.__int_size + 1):
            if (number // 2 ** j) % 2 == 1:
                for k in range(self.__int_size + 1 - j):
                    lam = - np.pi / (2 ** k) # We want to substract => minimization problem
                    #g = self.or_controlled_phase_gate(flags, lam)
                    phase_diff[self.cnf.get_num_vars() + j + k] += lam
                    #adder.append(g, [i for i in range(self.cnf.get_num_vars())] + [self.cnf.get_num_vars() + j + k])

        add_weights = QuantumCircuit(self.__int_size + 1)
        for key in phase_diff.keys():
            adder.p(-phase_diff[key], key)
            add_weights.p(phase_diff[key], key - self.cnf.get_num_vars())
        add_weights = add_weights.to_gate().control(num_ctrl_qubits=len(controls))
        adder.append(add_weights, controls + [i for i in range(self.cnf.get_num_vars(), self.cnf.get_num_vars() + self.__int_size + 1)])

        for i in range(len(flags)):
            if flags[i] == 1:
                adder.x(i)

        return adder.to_gate()

    def build_circuit(self):

        # Create a superposition of all possible assignments

        for i in range(self.cnf.get_num_vars()):
            self.circuit.h(i)

        # QFT of the integer registers

        transformer = QFT(self.__int_size + 1, do_swaps=False).to_gate()
        self.circuit.append(transformer, self.qr[-self.__int_size-2:-1])

        # Create a multi controlled gate for the clauses and add the weights, using flags

        for i in range(self.cnf.get_num_clauses()):

            flags = np.zeros(self.cnf.get_num_vars())
            for j in self.cnf.get_clauses()[i]:
                if j > 0:
                    flags[j - 1] = 1
                else:
                    flags[-j - 1] = -1

            # adder
            adder = self.adder_block(self.cnf.get_weights()[i], flags)
            self.circuit.append(adder, self.qr)

    def run(self):
        circuit = self.copy()
        cr = ClassicalRegister(self.__int_size + 1)
        circuit.add_register(cr)
        tr = QFT(self.__int_size + 1, do_swaps=False).to_gate().inverse()
        circuit.append(tr, self.qr[-self.__int_size-2:-1])
        for i in range(self.cnf.get_num_vars(), self.cnf.get_num_vars() + self.__int_size + 1):
            circuit.measure(i, i - self.cnf.get_num_vars())
        sampler = Sampler()
        results = sampler.run(circuit).result()
        statistics = results.quasi_dists[0].binary_probabilities()
        hist = plot_histogram(statistics)
        plt.ylabel("Prob. of measuring")
        plt.xlabel("Possible sum of weights")
        plt.title("Distribution of the sum of weights for all assignments of boolean variables")
        hist.savefig('../output/histogram_exact.png', bbox_inches="tight")

    def estimate(self, shots=1):
        circuit = self.copy()
        cr = ClassicalRegister(self.__int_size + 1)
        circuit.add_register(cr)
        tr = QFT(self.__int_size + 1, do_swaps=False).to_gate().inverse()
        circuit.append(tr, self.qr[-self.__int_size-2:-1])
        for i in range(self.cnf.get_num_vars(), self.cnf.get_num_vars() + self.__int_size + 1):
            circuit.measure(i, i - self.cnf.get_num_vars())
        estimator = Sampler()
        results = estimator.run(circuit, skip_transpilation=True, shots=shots).result()
        statistics = results.quasi_dists[0].binary_probabilities()
        hist = plot_histogram(statistics)
        hist.savefig('../output/histogram_estimated' + str(shots) + '.png', bbox_inches="tight")

    def draw(self, decomposition_level=0, show_info=False):
        c = self.circuit.copy()
        while decomposition_level > 0:
            c = c.decompose()
            decomposition_level -= 1
        if show_info:
            print("depth := " + str(c.depth()))
            print("size := " + str(c.size()))
        return c.draw()

    def get_circuit(self):
        return self.circuit

    def to_gate(self):
        return self.circuit.copy().to_gate()
    
    def copy(self):
        return self.circuit.copy()


if __name__ == "__main__":
    try:
        cnf_reader = CNFReader("../data/test2.cnf")
    except:
        print("Usage: python3 sat_circuit.py <cnf_file>")
        sys.exit(1)
    sat_circuit = SATCircuit(cnf_reader)
    display(sat_circuit.draw(decomposition_level=0, show_info=True))
    sat_circuit.run()
