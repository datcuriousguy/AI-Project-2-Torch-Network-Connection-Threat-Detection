import pandas as pd
import numpy as np

"""
These are the lists of possible values for the protocols, services, and flags of connections.
In other words, each element of the list is a type of protocol, service or flag respectively.
"""
protocols = ['TCP', 'UDP', 'ICMP']
services = ['HTTP', 'FTP', 'SSH', 'DNS', 'SMTP', 'Unknown']
flags = ['SF', 'S0', 'REJ', 'RSTO', 'S1', 'S2']


"""
This function generates each row of data for training. Since it is data for training, we already designate whether
the row contains data that is pertinent to:

- Benign connection request (0), or
- suspicious connection request (1).

1. is_malicious is randomly assigned with a higher probability of being 0 because 0 indicates benign (safe) and mos
   connection requests are benign.

2. protocol is one of the options 'HTTP', 'FTP', 'SSH', 'DNS', 'SMTP' or 'Unknown'.
   It is assigned by a uniform distribution of the probabolities 0.7, 0.2, and 0.1 over the 6 protocols specified in
   order to simulate realistic probability of getting a request from each specific type of protocol used. Again, the
   probabilities are meant to resemble reality.
   
3. flags are randomly chosen as one of 'SF', 'S0', 'REJ', 'RSTO', 'S1' and 'S2'.
   Essentially, the decreasing order of 'suspiciousness' is SF > S1 > S2 > S0 > REJ.
   
___

The if-else condition reflects if the random variable is_malicious is 0 or 1. 

(For this to be more clear, do take a look
at the SIGNIFICANCE OF THE DATA section within the Readme.md file for this project.)

If its malicious (i.e, 1):

    1. flag is more likely to be S0 or REJ
    
    2. src_bytes is abnormally low 50% of the time, and a value around 1000 (mean) bytes, otherwise.
    
    3. dst_bytes is more likely to be similar to src but adding 500 in order to simulate a response from the destination server.
    
    4. count is between 0 and 100 connection requests / second for suspicious connections (see count under the else statement)
    
    5. the serror_rate is a random decimal between 0.2 and 1 reflecting an incomplete connection rate between 20% and 100%,
       which would seem a reasonable percentage to flag as suspicious, rounded to two decimals
       
    6. rerror_rate is a random decimal between 0.2 and 1 reflecting a rejection (%age of connections that were actively rejected) rate
       between 10 and 100 percent, rounded to two decimals.
       
else (non malicious connection a.k.a 'benign'):

    1. flags are considered as 'SF', 'S1' or 'S2' which are relatively safer compared to S0 or REJ.
    
    2. as src_bytes is generally lower for safe requests, it is between 0 and 800 before being added to 200, ensuring a high likelihood of being
    below say, the src_bytes of a malicious connection request.
    
    3. The same idea applies to dst_bytes, being lower than the dst_bytes for a potentially malicious connection.
    
    4. count again, is lower (90% lower - between just 1 and 10) attempts / second for relatively safer
       connection requests.
       
    5/6. As the connection is safer, the serror_rate and rerror_rate must be lower by nature. Hence,
    here it is merely a fraction of the serror_rate and rerror_rate of a potentially unsafe connection request.
    
"""
def generate_row():
    is_malicious = np.random.choice([0, 1], p=[0.85, 0.15])
    protocol = np.random.choice(protocols, p=[0.7, 0.2, 0.1])
    service = np.random.choice(services, p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05])

    if is_malicious:
        flag = np.random.choice(['S0', 'REJ', 'RSTO'], p=[0.4, 0.4, 0.2])
        src_bytes = int(np.random.exponential(1000)) if np.random.rand() < 0.5 else 0
        dst_bytes = int(np.random.exponential(1000) + 500)
        count = np.random.randint(10, 100)
        serror_rate = round(np.random.uniform(0.2, 1.0), 2)
        rerror_rate = round(np.random.uniform(0.1, 1.0), 2)
    else:
        flag = np.random.choice(['SF', 'S1', 'S2'], p=[0.6, 0.2, 0.2])
        src_bytes = int(np.random.exponential(800) + 200)
        dst_bytes = int(np.random.exponential(800) + 100)
        count = np.random.randint(1, 10)
        serror_rate = round(np.random.uniform(0.0, 0.1), 2)
        rerror_rate = round(np.random.uniform(0.0, 0.1), 2)

    """
    we return a simple dictionary of each feature in a key-value pair. That's it! 
    Each row of our training dataset is ready to use.
    """

    return {
        "Protocol_type": protocol,
        "Service": service,
        "Flag": flag,
        "Src_bytes": src_bytes,
        "Dst_bytes": dst_bytes,
        "Count": count,
        "Serror_rate": serror_rate,
        "Rerror_rate": rerror_rate,
        "Malicious": is_malicious
    }

data = pd.DataFrame([generate_row() for _ in range(1000)])

"""
It appears that real world uses of data require it to be in a tabular form, which is why a dataframe seems like
a suitable datatype.
"""

print("\nSample Network Connection Training Data:\n")
print(data.head(len(data)).to_string(index=False))
print(f'\n\ntotal training data size: {len(data)}')   # <-- just for reference
