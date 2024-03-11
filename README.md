# NEAT-Based Agent
NEAT is an evolutionary algorithm that creates and optimizes neural network architectures. Unlike traditional neural networks, it not only adjusts the weights of the connections but also dynamically alters the network's structure. This process, which involves evolutionary algorithms for mutation and crossover, makes the network more adaptable to the task at hand.

## Genome Structure
The genome in this NEAT implementation encapsulates both the architecture and the weights of the neural network. Each genome is a sequence of genes, where each gene defines a layer in the neural network. The components of a gene are:
- Input and output sizes of the layer.
- Weights matrix: Dictates the strength of connections between neurons.
- Bias vector: Provides offsets for each neuron in the layer.
The structure of the genome can be visualized as a sequence of layers, with each layer being a gene. A simplified representation is as follows:

``` yaml
Genome: [Layer1, Layer2, ..., LayerN]
Layer: {
    Input Size,
    Output Size,
    Weights: [ [w11, w12, ...], [...], ... ],
    Bias: [b1, b2, ..., bN]
}
```
To prevent the creation of excessively large neural networks, limits were set on both the number of layers and the number of neurons per layer due to computational resource constraints.

## Neural Network Model
The neural network model is fully connected, meaning each neuron in a layer is connected to every neuron in the subsequent layer. This architecture ensures a thorough flow of information throughout the network.
```python
class Model(nn.Module):
    def __init__(self, genome):
        super(Model, self).__init__()

        # Create dinamically the intermediate layers based on the genome
        self.layers = nn.ModuleList()
        for i in range(len(genome)):
            input_size, output_size, weights, bias = genome[i]
            layer = nn.Linear(input_size, output_size)

            # Set weights and bias
            with torch.no_grad():
                layer.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
                layer.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

            self.layers.append(layer)

    def forward(self, x):
        # Forward pass through each intermediate layer
        for layer in self.layers:
            # Apply ReLU as activation function
            x = torch.relu(layer(x))

        # Apply softmax to the output layer
        return torch.softmax(x, dim=0)
```

## Simulation
During each generation, every individual neural network in the population competes against randomly selected opponents from the same group. This method ensures a variety of gameplay experiences, assessing each network's adaptability and strategy robustness. The individuals' performance is evaluated based on their success in these games, quantified as a fitness score.

## Mutation
The mutation process introduces random changes to the genome, affecting both the network's structure and its weights. The types of mutation include:
- Adding or removing a layer, increasing or decreasing network complexity.
    - **Layer Addition/Removal:** Layers can be randomly added between existing ones or removed.
- Adjusting the size of a layer by changing the number of neurons.
    - **Layer Resizing:** The number of neurons in a layer can be increased or decreased as necessary.
- Slight modifications to the existing weights and biases.
    - **Parameter Tweaking:** Individual weights and biases are subtly adjusted.

## Crossover
Crossover merges genes from two parent genomes to produce a new genome. This involves:
- Determining the number of layers in the child genome.
    - **Layer Numbers:** The child's layer count is randomly chosen, ranging from the parent with the fewest layers to the sum of both parents' layers minus the shared layers, namely input and output.
- Choosing layers from each parent, ensuring input and output size compatibility.
    - **Layer Selection:** A layer is randomly selected from one of the parents, respecting the layer order, and repeated until the child genome's layer count is reached.
- Combining weights and biases from the parents to create new layers.
    - **Parameter Mixing:** Weights and biases are blended from the chosen layers of both parents.

## Results
Unfortunately, with the hardware at my disposal, I was unable to train NEAT as I intended. I attempted to run the code for a few generations, limiting the number of individuals per generation, the games played per generation, and the complexity of the neural network. Despite all these constraints, I achieved interesting results, with a winning rate of 70% against a random opponent. These results are exciting, and it would be interesting to see how the agent performs if trained without all these limitations.