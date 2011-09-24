import random

def nutsAndBolts(nuts, bolts):
    assert len(nuts) == len(bolts)
    if len(nuts) == 0:
        return []
    elif len(nuts) == 1:
        return [(nuts[0][1], bolts[0][1])]
    chosen_n = random.choice(nuts)
    bolts_stn = []
    bolts_gtn = []
    for b in bolts:
        if b[0] < chosen_n[0]:
            bolts_stn.append(b)
        elif b[0] > chosen_n[0]:
            bolts_gtn.append(b)
        else:
            matching_b = b
    nuts_stb = []
    nuts_gtb = []
    for n in nuts:
        if n[0] < matching_b[0]:
            nuts_stb.append(n)
        elif n[0] > matching_b[0]:
            nuts_gtb.append(n)
    return [(chosen_n[1], matching_b[1])] + nutsAndBolts(nuts_stb, bolts_stn) + nutsAndBolts(nuts_gtb, bolts_gtn)
            
nuts = zip([10, 20, 30, 40, 50], range(0, 5))
bolts = zip([20, 30, 50, 40, 10], range(0, 5))
        
print nutsAndBolts(nuts, bolts)
