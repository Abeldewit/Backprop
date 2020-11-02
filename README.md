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
- `learn_rate`: this is the learning rate of the neural network
- `epochs`: this is how often the network will iterate over the training set. 
While running this 

When running the program the user is prompted to choose whether to run a GridSearch.
This GridSearch will try multiple options for the parameters mentioned above, or when the user declines using the 
GridSearch, a single run is done with the parameters that are (hard)coded in the main method of the program. 


#### After training
If you want to print the individual weights and bias weights of the program, you can do so by answering `print` to the `test:` question that is asked after
the network has finished training. 

If you wanted to visualize the weights of
the network after it has finished training, you can do so by answering `'vis'` to the `test:` question that is asked after 
the training. 