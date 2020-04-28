# Poseidon params generator (SAGE script)

c = 1
r = 2
Rf = 8
Rp = 83
n_rounds = Rf + Rp
m = r + c
p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
field = GF(p)
#generator of multiplicative subgroup
g = field(7)
IV = "POSEIDON"


def generate_round_constant(fn_name, field, idx):
    """
    Returns a field element based on the result of sha256.
    The input to sha256 is the concatenation of the name of the hash function
    and an index.
    For example, the first element for MiMC will be computed using the value
    of sha256('MiMC0').
    """
    from hashlib import sha256
    val = int(sha256('%s%d' % (fn_name, idx)).hexdigest(), 16)
    return field(val)
    

def generate_mds_matrix(name, field, m, num_attempts = 100):
    """
    Generates an MDS matrix of size m x m over the given field, with no
    eigenvalues in the field.
    Given two disjoint sets of size m: {x_1, ..., x_m}, {y_1, ..., y_m} we set
    A_{ij} = 1 / (x_i - y_j).
    """
    for attempt in xrange(100):
        x_values = [generate_round_constant(name + 'x', field, attempt * m + i)
                    for i in xrange(m)]
        y_values = [generate_round_constant(name + 'y', field, attempt * m + i)
                    for i in xrange(m)]
        
        # Make sure the values are distinct.
        if len(set(x_values + y_values)) != 2 * m:
            continue
        mds = matrix([[1 / (x_values[i] - y_values[j]) for j in xrange(m)]
                      for i in xrange(m)])
        
        if len(mds.characteristic_polynomial().roots()) == 0:
            # There are no eigenvalues in the field.
            return mds
    raise Exception('No good MDS found')
                                            
                                            
def generate_parameters(field, m, n_rounds, iv):
    
    ark = [vector(generate_round_constant(iv, field, m * i + j) for j in xrange(m)) for i in xrange(n_rounds)]
    mds = generate_mds_matrix('iv', field, m)
    
    return (ark, mds)

                                            
ark, mds = generate_parameters(field, m, n_rounds, IV)

def pretty_hex(num):
    rep = hex(num)
    rep = "0x" + "0" * (8 - len(rep)) + rep
    return rep


def field_pretty_printer(elem):
    elem = int(elem)
    modulus = 2^(32)
    res = "{ "
    for i in xrange(8):
        cur = elem % modulus
        elem = elem >> 32
        res += pretty_hex(cur)
        if i < 7:
            res += ", "
        
    res += " }"
    return res
    
        
print "ARK"

for (i, round) in enumerate(ark):
    for idx, elem in enumerate(round):
        if idx == 0:
            print "{\n\t" + field_pretty_printer(elem) + ", "
        elif idx == (r+c-1):
            print "\t" + field_pretty_printer(elem) + "\n},\n"
        else:
            print "\t" + field_pretty_printer(elem) + ","
            
            
print "MDS"

for (i, round) in enumerate(mds):
    for idx, elem in enumerate(round):
        if idx == 0:
            print "{\n\t" + field_pretty_printer(elem) + ", "
        elif idx == (r+c-1):
            print "\t" + field_pretty_printer(elem) + "\n},\n"
        else:
            print "\t" + field_pretty_printer(elem) + ","