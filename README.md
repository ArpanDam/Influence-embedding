# Influence-embedding
This code is for the paper titled "On the Badge Aware Influence Representation Learning & Influencer Detection in Social Networks".

network.py is the multitask learning model. It is trained for each badge and the saved model for each badge is stored in the folder "saved model".

probability_of_influence.py file calculates the badge wise probality of influence between the influencer and follower.
it applies the sigmoid function on the dot product of influencer and follower embedding and calculates the probality of influence and store the badge wise probablity in the
file "dict_probability_of_influence".

seed_finder.py finds the k best influencers.
tag_finder.py finds the r best badges for the k best influencer.

