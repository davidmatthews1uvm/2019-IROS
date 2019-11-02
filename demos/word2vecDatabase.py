from __future__ import print_function

import os
import sys
import time
from builtins import input

import sqlite3 as lite
import pickle

class Word2VecVectorSpace(object):
    def __init__(self, database_file):
        self.database_file = database_file
        self.con = lite.connect(database_file)
        self.cur = self.con.cursor()

    def get_vector(self, word):
        """
        Searches for the vector representing the given string

        :param word: String to locate vector for
        :return: Numpy array of the vector corresponding to the word.
        If no corresponding vector is found, rasies a KeyError Exception
        ;rtype: numpy.ndarray
        """
        string = "SELECT * FROM Vectors WHERE name=?"
        params = (word,)
        self.cur.execute(string, params)
        raw_vector = self.cur.fetchone()
        if raw_vector is None:
            raise KeyError("Vector not found")
        else:
            vector = pickle.loads(raw_vector[1])
        return vector

    def build_database(self, w2v_file, print_progress=False):
        from gensim.models import KeyedVectors
        def add_vector(name, vec):
            self.cur.execute("insert into Vectors values (?,?)", (name, pickle.dumps(vec, protocol=0)))
            self.con.commit()

        def reset_database():
            self.cur.execute("drop table if exists Vectors")
            self.cur.execute("create table Vectors(name TEXT, vector BLOB)")

        reset_database()
        word_vectors = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        numb = 0
        if print_progress: print(len(word_vectors.vocab))
        for key in word_vectors.vocab:
            if numb % 100 == 0 and print_progress:
                print(numb, "\t\t", sep="", end="")
                sys.stdout.flush()
            numb += 1
            metadata = word_vectors.vocab[key]
            vec = word_vectors.vectors[metadata.index]
            add_vector(key, vec)

if __name__ == "__main__":
    from scipy import spatial

    db = Word2VecVectorSpace(database_file='w2vVectorSpace-google.db')
    cmd = input("Please type a command: EXIT, FIND, COMPARE: ")
    while cmd.rstrip() != "EXIT":
        if cmd == "FIND" or cmd == "F":
            word = input("Please type a word to look for: ")
            try:
                db.get_vector(word)
            except KeyError:
                print("Vector Not Found")
            else:
                print("Vector found!")
        elif cmd == "COMPARE" or cmd == "C":
            w1 = input("Please type a word1: ")
            try:
                v1 = db.get_vector(w1)
            except KeyError:
                print("Vector Not Found")

            else:
                w2 = input("Please type a word2: ")
                try:
                    v2 = db.get_vector(w2)
                except KeyError:
                    print("Vector Not Found")
                else:
                    print("Cos similarity is: ", 1 - spatial.distance.cosine(v1,v2))
        elif cmd == "ADD" or cmd == "A":
            w1 = input("Please type a word1: ")
            try:
                v1 = db.get_vector(w1)
            except KeyError:
                print("Vector Not Found")

            else:
                w2 = input("Please type a word2: ")
                try:
                    v2 = db.get_vector(w2)
                except KeyError:
                    print("Vector Not Found")
                else:
                    w3 = input("Please type a word3: ")
                    try:
                        v3 = db.get_vector(w3)
                    except KeyError:
                        print("Vector Not Found")
                    else:
                        print("Cos similarity is: ", 1 - spatial.distance.cosine(v1+v2, v3))
        elif cmd == "PRINT" or cmd == "P":
            w1 = input("Please type a word: ")
            try:
                v1 = db.get_vector(w1)
            except KeyError:
                print("Vector Not Found")

            else:
                print(list(v1))

        cmd = input("Please type a command: EXIT, FIND, COMPARE: ")

