# This file contains the class CNFReader, which is used to read CNF files for the weighted max-sat problem.

import re


class CNFReader:

    def __init__(self, file_path):
        self.file_path = file_path
        self.__num_vars = 0
        self.__num_clauses = 0
        self.__clauses = []
        self.__weights = []
        self.read_cnf_file()

    def read_cnf_file(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith('c'):
                    continue
                elif line.startswith('p'):
                    self.__num_vars, self.__num_clauses = map(int, re.findall(r'\d+', line))
                else:
                    self.__clauses.append(list(map(int, line.split()[:-1])))
                    self.__weights.append(int(line.split()[-1]))

    def get_num_vars(self):
        return self.__num_vars

    def get_num_clauses(self):
        return self.__num_clauses

    def get_clauses(self):
        return self.__clauses

    def get_weights(self):
        return self.__weights


if __name__ == '__main__':
    cnf_reader = CNFReader('../data/test.cnf')
    print(cnf_reader.get_num_vars())
    print(cnf_reader.get_num_clauses())
    print(cnf_reader.get_clauses())
    print(cnf_reader.get_weights())
