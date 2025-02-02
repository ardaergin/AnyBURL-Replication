import os
import pickle
from algorithm.knowledge_graph import Triple, KnowledgeGraph
from algorithm.path_sampling import BottomRule, sample_bottom_rule
from algorithm.rule_generalization import GeneralizedRule, generalize_bottom_rule

# Example
print("\n\n===== Dataset: WN18RR =====")
with open("data/triples/WN18RR_triples_train.pkl", "rb") as dataset:
    WN18RR_train_KG_dataset = pickle.load(dataset)
WN18RR_train_KG = KnowledgeGraph(WN18RR_train_KG_dataset)

# Example bottom rules
print("\n--- Example: (any) bottom rules ---")
example_bottom_rule = sample_bottom_rule(WN18RR_train_KG, n=2)
print(example_bottom_rule)
example_bottom_rule = sample_bottom_rule(WN18RR_train_KG, n=3)
print(example_bottom_rule)
example_bottom_rule = sample_bottom_rule(WN18RR_train_KG, n=4)
print(example_bottom_rule)
example_bottom_rule = sample_bottom_rule(WN18RR_train_KG, n=5)
print(example_bottom_rule)

# Example cyclical bottom rule
print("\n--- Example: cyclical rule ---")
example_cyclical_bottom_rule = None
while True:
    example_cyclical_bottom_rule = sample_bottom_rule(WN18RR_train_KG, n=4)
    if example_cyclical_bottom_rule is None:
        continue
    if example_cyclical_bottom_rule.is_cyclical:
        print(example_cyclical_bottom_rule)
        break

print("\n\n===== Rule Generalization =====")
print("\n--- Example (any) bottom rule generalization ---")
generalized_rules_1 = generalize_bottom_rule(example_bottom_rule)
for rules in generalized_rules_1: 
    print(rules)
print("\n--- Example cyclical rule generalization ---")
generalized_rules_2 = generalize_bottom_rule(example_cyclical_bottom_rule)
for rules in generalized_rules_2: 
    print(rules)

print("\n")
