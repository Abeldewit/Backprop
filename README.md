# Backprop
_ACML Assignment_

**Requirements:**
- Python 
- Pyplot ( if visualization of the weights is wanted )

For this assignment we are tasked to write our own 
augmented neural network using the knowledge about neuron activation, the sigmoid function, and
forward- and back-propagation. 


### Run instructions
To run this program, all you need to do is run the main method of `main.py`. 

There are two parameters in the main method to play around with:
- `rates`: this is the learning rate of the neural network
- `iterations`: this is how often the network will iterate over the training set
- `decays`: this are the weight decays

When running the program the user is prompted to choose whether to run a GridSearch.
This GridSearch will try multiple options for the parameters mentioned above, or when the user declines using the 
GridSearch, a single run is done with the optimal parameters that are (hard)coded in the main method of the program. 


#### After training
The program prints the final error rate of all parameters and produces a plot of the error rates throughout all
 iterations. 

If a single run of the program was chosen by the user, the network and it's weights will be visualized.
