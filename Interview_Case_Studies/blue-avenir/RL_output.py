from random import choices as rl_agent_decisioning
import numpy as np

def rl_agent(customer_state, offers):
    weights = []
    print(f"logging customer_state age: {customer_state['age']}" )
    if customer_state["age"] < 30:
        weights = [10, 8 ,6, 6, 6, 4]
    else:
        weights = [8 ,6, 10, 6, 4]

    print(f"logging customer_state region: {customer_state['region']}")
    if customer_state['region'] == "Europa":
       weights = [10, 8 ,6, 6, 10, 10]
    elif customer_state['region'] == "Americas":
       weights = [4, 8 ,6, 6, 10, 10]
    elif customer_state['region'] == "Asia":
       weights = [4, 10 , 10, 6, 4, 4]
    else:
       raise Exception(f"weight by region not applied, is categorical data ?")

    print(f"logging customer_state gender: {customer_state['gender']}" )
    if customer_state['gender'] == "M":
       weights = [10, 10 , 10, 6, 4, 4]
    elif customer_state['gender'] == "F":
       weights = [4, 4, 6, 10, 10, 10]
    else:
       raise Exception(f"weight by gender was not applied, is categorical data?")

    # the module does not accepts dict only list
    offers = [i for i in offers.keys()]

    # the len needs to be matched at any case
    if len(offers) > len(weights):
        n = len(offers) - len(weights)
        for i in range(n):
           weights.append(np.random.choice(n))

    return rl_agent_decisioning(offers, weights=weights[:len(offers)] , k = 3)