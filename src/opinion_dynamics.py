import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bounded_confidence_model(eps, opinions, opinion_i):
    """
    :param eps: Bounded Confidence Treshold
    :param opinions: Set of considered Opinions
    :param opinion_i: Opinion of i at t
    :return: new opinion
    """

    valid_opinions = opinions[(np.abs(opinions - opinion_i) <= eps)]
    if len(valid_opinions) > 0:
        #new_opinion = np.mean(np.append(valid_opinions, opinion_i))
        new_opinion = np.mean(valid_opinions)
    else:
        new_opinion = opinion_i

    return new_opinion


def one_step(opinions, eps, political_messages):
    new_opinions = []
    for agent in opinions:
        #messages = np.random.uniform(0,1, 10)
        messages = np.random.choice(political_messages, size=10, replace=True)
        new_opinions.append(bounded_confidence_model(eps=eps, opinions=messages, opinion_i=agent))

    return np.array(new_opinions)


def simulation(n_agents, eps, political_messages, t=50):
    results = []
    #opinion_at_t = np.random.uniform(0, 1, n_agents)
    opinion_at_t = np.random.beta(2, 2, size=n_agents)
    for _ in range(t):
        results.append(opinion_at_t)
        opinion_at_t = one_step(opinion_at_t, eps, political_messages)

    return np.column_stack(results)



def plot_opinions(opinion_vector):
    histo = []
    bins = np.linspace(0,1, num=21)
    for i in range(opinion_vector.shape[1]):
        histo.append(np.histogram(opinion_vector[:, i], bins=bins)[0])

    df = pd.DataFrame(np.column_stack(histo))
    df_normalized = df / df.sum()
    # make plot
    plt.figure(figsize=(20, 10))
    plt.imshow(df_normalized, aspect='auto', cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel('Operation Index')
    plt.ylabel('Data Point Index')
    plt.title('Heatmap of Operations')
    plt.show()


