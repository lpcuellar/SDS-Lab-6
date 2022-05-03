import  numpy as np
import  re

def H_entropy (x):
    # Entropia de Shannon
    prob = [ float(x.count(c)) / len(x) for c in dict.fromkeys(list(x)) ] 
    H = - sum([ p * np.log2(p) for p in prob ]) 
    return H

def posicionPrimerDigito( s ):
    for i, c in enumerate(s):
        if c.isdigit():
            return i + 1
    return 0

def proporcionVocalesConsonantes (x):
    x = x.lower()
    patron_vocales = re.compile('([aeiou])')
    patron_consonantes = re.compile('([b-df-hj-np-tv-z])')
    vocales = re.findall(patron_vocales, x)
    consonantes = re.findall(patron_consonantes, x)
    try:
        proporcion = len(vocales) / len(consonantes)
    except: # Division por cero
        proporcion = 0  
    return proporcion