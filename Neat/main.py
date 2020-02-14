#%%
import gym

from Neat.structs import (
    NeuralNetwork,
    InputNode,
    OutputNode,
    Innovations,
)

#%%
# parameters
environment_name = "CartPole-v0"
initial_network_amount = 10


#%%
# generate environments
environments = [gym.make(environment_name)]

#%%
# get input and output amounts from dummy environment
dummy_env = gym.make(environment_name)
input_amount = dummy_env.observation_space.shape[0]
output_amount = dummy_env.action_space.n

#%%
# generate global innovation history
global_innovation_history = Innovations()
input_nodes = []
output_nodes = []

# generate global input and output nodes
for i in range(input_amount):
    input_nodes.append(InputNode(global_innovation_history.max_node_innovation_id))
    global_innovation_history.max_node_innovation_id += 1

for i in range(output_amount):
    output_nodes.append(OutputNode(global_innovation_history.max_node_innovation_id, 0))
    global_innovation_history.max_node_innovation_id += 1

#%%
# generate initial networks using input and output nodes
networks = []
for i in range(initial_network_amount):

    # copy global innovations into network innovations
    network_innovations = Innovations()

    # nodes
    network_innovations.max_node_innovation_id = (
        global_innovation_history.max_node_innovation_id
    )
    network_innovations.node_innovations = global_innovation_history.node_innovations[:]

    # connections
    network_innovations.max_connection_innovation_id = (
        global_innovation_history.max_connection_innovation_id
    )
    network_innovations.connection_innovations = global_innovation_history.connection_innovations[
        :
    ]

    # generate new network
    networks.append(
        NeuralNetwork(
            input_nodes=input_nodes[:],
            hidden_nodes=[],
            output_node=output_nodes[:],
            connections=[],
            innovations=network_innovations,
        )
    )
