<hr>
<h2 id="the-situation">The Situation</h2>
<p>We are working in a company where a large number of network requests and connections are received. It may be a Cybersecurity Firm, a Financial Management Firm, A Large Bank, or any large firm.</p>
<h2 id="the-problem">The Problem</h2>
<p>Due to the nature of our large literal and metaphorical network, there is always a chance that we could receive a network connection request with malicious intent. Such requests may have the intent of causing a DDoS (Distributed Denial of Service) attack,  hacker’s attempt to break into some component of the system or the servers, or just Malware.</p>
<h2 id="dealing-with-the-problem">Dealing with The Problem</h2>
<p>We know that connection attempts found to be malicious in the past have shown some common features, such as:</p>
<ul>
<li>High number of connection requests within a relatively short time frame</li>
<li>High number of connection requests within a relatively short time frame with rejections</li>
<li>High rejection rates for a connection</li>
<li>Large number of ICMP (Internet Control Message Protocol) requests. Though ICMP is traditionally used for error reports and network diagnostics, but its ability to hide data within ICMP Packets, along with other characteristics, make it an attractive choice for attackers.</li>
<li>History of connection attempts being rejected hinting at possible ‘guessing’ login attempts.</li>
<li>Unusual number of bytes of data being sent of received - could indicate data exfiltration or pinging which could both indicate suspicious activity</li>
<li>Insecure or outdated ports used by attackers to avoid detection.</li>
<li>Port numbers that are not commonly used can also hint at suspicious activity attempts.</li>
</ul>
<p>Note that detecting multiple features as odd accentuates the suspiciousness of a connection attempt.</p>
<p>Files:</p>
<ul>
<li><p><strong><code>Readme.md</code></strong></p>
<p>  <strong>This file.</strong></p>
</li>
<li><p><code>sample_data.md</code></p>
<p>  A sample of training data to be used to train the model.</p>
</li>
<li><p><strong><code>model.py</code></strong></p>
<p>  The python code for the neural network alone, including hardcoded hyperparameters.</p>
</li>
<li><p><strong><code>resources.md</code></strong></p>
<p>  A text file containing some resources on data for training and the significance of each of the features used to train the model.</p>
</li>
<li><p><code>database_control.py</code></p>
<p>  This is a python file whose sole purpose is to create and then make manipulations to the MySQL database which will store our training data - which is also created in this file.</p>
</li>
<li><p><code>requirements.md</code></p>
<p>  Lists the libraries required to run the neural network and get a desired output.</p>
</li>
<li><p>Database information (Local MySQL Database):</p>
<ul>
<li>Username: <code>daniel</code></li>
<li>Password: <code>daniel</code></li>
<li>Database name: <code>nn_project_2_database</code></li>
</ul>
</li>
</ul>
<hr>
<h2 id="data-source">Data Source</h2>
<p>I have used <code>database_control.py</code> to: </p>
<ul>
<li>Create the data to be used for training the neural network. I have made sure the parameters (see docstrings) are as realistic as possible.</li>
<li>Load the data into a local MySQL Database (The purpose of doing it like this is to try and simulate how data is stored prior to training models in real world business scenarios)</li>
</ul>
<hr>
<h2 id="understanding-the-significance-of-the-data">Understanding the Significance of the Data</h2>
<p>A helpful source of info for this: <a href="https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/">https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/</a></p>
<ol>
<li><p>Protocol Type:</p>
<p> This denotes what protocol has been used for a connection request. Some examples are</p>
<ul>
<li>TCP (Transmission control Protocol)</li>
<li>UDP (User Datagram Protocol)</li>
<li>ICMP (Internet Control Message Protocol)</li>
</ul>
<p> In general, ICMP is a common choice for scanning or ping-attacks, however requests sent by the relatively common TCP/UDP protocols can still be seen to contain odd behaviour. Even though a connection request can be suspicious with malicious attempt, regardless of protocol, it is still a factor worth considering as number of requests taken into account <em><strong>alongside</strong></em> protocol can hold more weight leaning towards a suspicious request.
 </p>
</li>
<li><p>Service:</p>
<p> This is the network service used by the destination port used for the given connection. Common service types or methods include the well-known</p>
<ul>
<li>HTTP</li>
<li>HTTPS</li>
</ul>
<p> As well as some lesser known ones like</p>
<ul>
<li>FTP</li>
<li>Telnet</li>
</ul>
<p> Hence, rare or unknown services used for a connection can hint at scanning (pinging with malicious intent) or use of some customized malware. Perhaps on the security side, we could say HTTPS &gt; HTTP &gt; FTP &gt; Telnet, or something like that. 
 </p>
</li>
<li><p>Flag Used:</p>
<p> This refers to the status flag of the connection, and can include:</p>
<ul>
<li>S0 - Means no response from the target server was received.</li>
<li>REJ - Indicates a rejected connection request.</li>
<li>SF - Safe, ‘normal’ traffic, or what we could refer to as ‘benign’ in this networking context.</li>
</ul>
</li>
<li><p>Src_bytes or Source Bytes:</p>
<p> The number of bytes of the data being sent from cloient to server. We are looking for abnormally low or abnormally high number of bytes. </p>
<ul>
<li>Abnormally Low: Indicates pinging or scanning attempts</li>
<li>Abnormally High: Indicates attempts to data theft or <strong>data exfiltration</strong> as it is called in cybersecurity.</li>
</ul>
</li>
<li><p>Dst_bytes or Destination Bytes:</p>
<p> Simply the opposite of src_bytes - it is the number of bytes sent from the destination back to the initial source in return.</p>
<ul>
<li>Abnormally High: May indicate a DDoS or overloading / spam attack</li>
<li>Abnormally Low: May indicate scanning or pinging (hmm is it safe to hack this guy’s system??)</li>
</ul>
</li>
<li><p>Count (of connections):</p>
<p> Number of connections to the same host in a given time duration (usually in seconds). What we need to know: &gt; 50 may indicate pinging, brute force attacks, or again, DDoS attacks.</p>
</li>
<li><p>Serror_rate (half-open attempts to connect):</p>
<p> A high Serror_rate may indicate repeated failed connections, which may be caused by pinging (as mentioned before), or attacks on (rightfully) closed ports.</p>
</li>
<li><p>Rerror_rate (same host):</p>
<p> The  percentage of connection attempts to the same host that were rejected. If over 50% of connection attempts were rejected, it indicates unautorized access attempts, which are suspicious</p>
</li>
<li><p>Malicious:</p>
<p> This can be either 0 or 1. </p>
<ul>
<li>0 Indicates benign connection attempt (safe).</li>
<li>1 indicates possibly malicious connection attempt.</li>
</ul>
</li>
</ol>
<hr>
<h2 id="how-the-neural-network-is-trained">How The Neural Network is Trained</h2>
<ol>
<li><p>Forward Pass to get predictions</p>
</li>
<li><p>Compute Loss of those predictions</p>
</li>
<li><p>Backpropagage to compute gradients </p>
<p> (Find out how much the error changes for a given change in gradient</p>
</li>
<li><p>Recompute weights </p>
<p> (New weight = Old weight – Learning rate × Gradient)</p>
</li>
<li><p>Repeat for all batches, if any</p>
</li>
</ol>
<hr>
<h2 id="best-practices-while-creating-the-neural-network-from-other-github-projects">Best Practices while Creating The Neural Network (From other GitHub Projects)</h2>
<ol>
<li>Using a Modular Architecture by making use of layers, modular functions, feed-forward and feed-backward functions, loss-functions, and optimizers. This is good for readability and makes things reusable as well.</li>
<li>Defining and implementing ways to handle missing values, normalizing features (if needed), and taking note of outliers are all recommended practices in the GitHub Community. Data cleaning and preprocessing prior to use for training will make sure that the neural network is properly trained while avoiding or reducing the chance for anomalous results.</li>
<li>Using regularization (minimizing needless complexity and exposing the network to more diverse data) will help prevent overfitting while improving generalization, which is desired in most neural networks. Note: There are multiple methods for regularization.</li>
<li>Optional: Tuning Hyperparameters like epochs and learning rate specific to strategies such as grid search or Beayesian optimization, can enhance model performance.</li>
<li>Taking note of the Bias and Variance of a given model is important as it enables the tuning of hyperparameters to match different scenarios (Essentially finding a point between Linear Regression and overfitting).</li>
<li>Obviously, a well-written README file (hello there) and good directory structure are useful to have for any given project.</li>
</ol>
<hr>
<h2 id="sample-output">Sample Output</h2>
<p><code>Epoch [10/100], Loss: 0.6388 Epoch [20/100], Loss: 0.5974 Epoch [30/100], Loss: 0.5488 Epoch [40/100], Loss: 0.4905 Epoch [50/100], Loss: 0.4185 Epoch [60/100], Loss: 0.3344 Epoch [70/100], Loss: 0.2498 Epoch [80/100], Loss: 0.1783 Epoch [90/100], Loss: 0.1242 Epoch [100/100], Loss: 0.0855</code></p>
<p><code>Test Accuracy: 99.75% Connection 1: 🟢 Normal Connection 2: 🔴 Suspicious Connection 3: 🔴 Suspicious</code></p>
<p><code>Process finished with exit code 0</code></p>
