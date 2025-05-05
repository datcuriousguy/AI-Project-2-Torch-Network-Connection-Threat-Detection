# Torch-NN-Project-2-Network-Connection-Threat-Detection
This project simulates a large organization's network, which regularly faces malicious connection attempts like DDoS, hacking, or malware. These threats often share traits such as high request rates, unusual ports, and frequent rejections. We want to design a Torch Neural Network that detects these suspicious connection attempts.

---


## The Situation

We are working in a company where a large number of network requests and connections are received. It may be a Cybersecurity Firm, a Financial Management Firm, A Large Bank, or any large firm.

## The Problem

Due to the nature of our large literal and metaphorical network, there is always a chance that we could receive a network connection request with malicious intent. Such requests may have the intent of causing a DDoS (Distributed Denial of Service) attack,  hacker’s attempt to break into some component of the system or the servers, or just Malware.

## Dealing with The Problem

We know that connection attempts found to be malicious in the past have shown some common features, such as:

- High number of connection requests within a relatively short time frame
- High number of connection requests within a relatively short time frame with rejections
- High rejection rates for a connection
- Large number of ICMP (Internet Control Message Protocol) requests. Though ICMP is traditionally used for error reports and network diagnostics, but its ability to hide data within ICMP Packets, along with other characteristics, make it an attractive choice for attackers.
- History of connection attempts being rejected hinting at possible ‘guessing’ login attempts.
- Unusual number of bytes of data being sent of received - could indicate data exfiltration or pinging which could both indicate suspicious activity
- Insecure or outdated ports used by attackers to avoid detection.
- Port numbers that are not commonly used can also hint at suspicious activity attempts.

Note that detecting multiple features as odd accentuates the suspiciousness of a connection attempt.

Files:

- **`Readme.md`**
    
    **This file.**
    
- `sample_data.md`
    
    A sample of training data to be used to train the model.
    
- **`model.py`**
    
    The python code for the neural network alone, including hardcoded hyperparameters.
    
- **`resources.md`**
    
    A text file containing some resources on data for training and the significance of each of the features used to train the model.
    
- `database_control.py`
    
    This is a python file whose sole purpose is to create and then make manipulations to the MySQL database which will store our training data - which is also created in this file.
    
- `requirements.md`
    
    Lists the libraries required to run the neural network and get a desired output.
    
- Database information (Local MySQL Database):
    - Username: `root`
    - Password: `password`
    - Database name: `nn_project_2_database`

---

## Data Source

I have used `database_control.py` to: 

- Create the data to be used for training the neural network. I have made sure the parameters (see docstrings) are as realistic as possible.
- Load the data into a local MySQL Database (The purpose of doing it like this is to try and simulate how data is stored prior to training models in real world business scenarios)

---

## Understanding the Significance of the Data

A helpful source of info for this: https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/

1. Protocol Type:
    
    This denotes what protocol has been used for a connection request. Some examples are
    
    - TCP (Transmission control Protocol)
    - UDP (User Datagram Protocol)
    - ICMP (Internet Control Message Protocol)
    
    In general, ICMP is a common choice for scanning or ping-attacks, however requests sent by the relatively common TCP/UDP protocols can still be seen to contain odd behaviour. Even though a connection request can be suspicious with malicious attempt, regardless of protocol, it is still a factor worth considering as number of requests taken into account ***alongside*** protocol can hold more weight leaning towards a suspicious request.
    
2. Service:
    
    This is the network service used by the destination port used for the given connection. Common service types or methods include the well-known
    
    - HTTP
    - HTTPS
    
    As well as some lesser known ones like
    
    - FTP
    - Telnet
    
    Hence, rare or unknown services used for a connection can hint at scanning (pinging with malicious intent) or use of some customized malware. Perhaps on the security side, we could say HTTPS > HTTP > FTP > Telnet, or something like that. 
    
3. Flag Used:
    
    This refers to the status flag of the connection, and can include:
    
    - S0 - Means no response from the target server was received.
    - REJ - Indicates a rejected connection request.
    - SF - Safe, ‘normal’ traffic, or what we could refer to as ‘benign’ in this networking context.
4. Src_bytes or Source Bytes:
    
    The number of bytes of the data being sent from cloient to server. We are looking for abnormally low or abnormally high number of bytes. 
    
    - Abnormally Low: Indicates pinging or scanning attempts
    - Abnormally High: Indicates attempts to data theft or **data exfiltration** as it is called in cybersecurity.
5. Dst_bytes or Destination Bytes:
    
    Simply the opposite of src_bytes - it is the number of bytes sent from the destination back to the initial source in return.
    
    - Abnormally High: May indicate a DDoS or overloading / spam attack
    - Abnormally Low: May indicate scanning or pinging (hmm is it safe to hack this guy’s system??)
6. Count (of connections):
    
    Number of connections to the same host in a given time duration (usually in seconds). What we need to know: > 50 may indicate pinging, brute force attacks, or again, DDoS attacks.
    
7. Serror_rate (half-open attempts to connect):
    
    A high Serror_rate may indicate repeated failed connections, which may be caused by pinging (as mentioned before), or attacks on (rightfully) closed ports.
    
8. Rerror_rate (same host):
    
    The  percentage of connection attempts to the same host that were rejected. If over 50% of connection attempts were rejected, it indicates unautorized access attempts, which are suspicious
    
9. Malicious:
    
    This can be either 0 or 1. 
    
    - 0 Indicates benign connection attempt (safe).
    - 1 indicates possibly malicious connection attempt.

---

## How The Neural Network is Trained

1. Forward Pass to get predictions
2. Compute Loss of those predictions
3. Backpropagage to compute gradients 
    
    (Find out how much the error changes for a given change in gradient
    
4. Recompute weights 
    
    (New weight = Old weight – Learning rate × Gradient)
    
5. Repeat for all batches, if any

---

## Best Practices while Creating The Neural Network (From other GitHub Projects)

1. Using a Modular Architecture by making use of layers, modular functions, feed-forward and feed-backward functions, loss-functions, and optimizers. This is good for readability and makes things reusable as well.
2. Defining and implementing ways to handle missing values, normalizing features (if needed), and taking note of outliers are all recommended practices in the GitHub Community. Data cleaning and preprocessing prior to use for training will make sure that the neural network is properly trained while avoiding or reducing the chance for anomalous results.
3. Using regularization (minimizing needless complexity and exposing the network to more diverse data) will help prevent overfitting while improving generalization, which is desired in most neural networks. Note: There are multiple methods for regularization.
4. Optional: Tuning Hyperparameters like epochs and learning rate specific to strategies such as grid search or Beayesian optimization, can enhance model performance.
5. Taking note of the Bias and Variance of a given model is important as it enables the tuning of hyperparameters to match different scenarios (Essentially finding a point between Linear Regression and overfitting).
6. Obviously, a well-written README file (hello there) and good directory structure are useful to have for any given project.

---

## Sample Output
