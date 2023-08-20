# chaotic logistic map for generating keys

def keygen(x, r, size):
    key = []

    for i in range(size):
        x = x * r * (1 - x)  # logistic map

        key.append(int((x * pow(10,16)) % 256))  # key = (x*10^16)%256

    return key


#print(keygen(0.3894, 3.9159, 10000))
