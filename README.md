# Influence-embedding
This code is implementation of Badge2vec for the paper titled "On the Badge Aware Influence Representation Learning & Influencer Detection in Social Networks".

network.py is the multitask learning model. It is trained for each badge and the saved model for each badge is stored in the folder "saved model".

probability_of_influence.py file calculates the badge wise probality of influence between the influencer and follower.
it applies the sigmoid function on the dot product of influencer and follower embedding and calculates the probality of influence and store the badge wise probablity in the
file "dict_probability_of_influence".

seed_finder.py finds the k best influencers.
tag_finder.py finds the r best badges for the k best influencer.

Run `python tag_finder.py' to find top 5 influencial members and top 2 influence badges.

Run `python tag_finder.py number_of_influencial_users number_of_influence_tags' to get the output as top-k influencial nodes and top-r influence badges.

To train the model again -> Run `network.py'

To complete the entire process of training the model, finding probablity of influence and find the top influencers and top badges - Run python network.py && python probability_of_influence.py && python tag_finder.py


