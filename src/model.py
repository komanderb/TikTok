import numpy as np

class Videos():
    def __init__(self, n_videos, parameters):
        self.video_parameters = parameters
        self.n_videos = n_videos


class Users(Videos):
    def __init__(self, n_agents: int, parameters: dict, n_videos: int, video_parameters: dict):
        super().__init__(n_videos=n_videos, parameters=video_parameters)
        self.n_agents = n_agents
        self.user_parameters = parameters

    def opinion_update(self):
        opinion_dif = self.user_parameters["considered_opinions"] - self.user_parameters["opinion"]

        t_opinion = self.user_parameters['opinion'] + (self.user_parameters['lambda'] * opinion_dif)
        self.user_parameters['opinion'] = t_opinion
        #TODO: what is the best format here?
        #TODO: fix nan's here;



class RecSyS(Users):
    def __init__(self, rec_parameters, user_parameters,
                 video_parameters, n_agents, n_videos, cold_start=10):
        super().__init__(parameters=user_parameters, video_parameters=video_parameters,
                         n_agents=n_agents, n_videos=n_videos)
        self.parameters = rec_parameters
        self.cold_start = cold_start

    def initialize_recsys(self):
        matrix = np.zeros((self.n_agents, self.n_videos), dtype=int)
        for col in range(self.n_videos):
            row_indices = np.random.choice(self.n_agents, self.cold_start, replace=False)
            matrix[row_indices, col] = 1

        self.parameters['current_recommendation'] = matrix
        self.parameters['all_recommendations'] = matrix
        self.parameters["all_interactions"] = np.zeros((self.n_agents, self.n_videos))
        self.parameters["position_matrix"] = np.ones((self.n_agents, self.n_videos))

    def get_popularity(self):
        popularity = np.sum(self.parameters['all_interactions'], axis=0)
        total_popularity = np.sum(popularity)
        normalized_popularity = popularity / total_popularity
        return normalized_popularity

    def get_user_position(self):
        for user in range(self.n_agents):
            interacted_videos = self.parameters['all_interactions'][user, :]
            interacted_political = np.logical_and(interacted_videos == 1, self.video_parameters["political"] == 1)
            interacted_other = np.logical_and(interacted_videos == 1, self.video_parameters['political'] == 0)
            political_idx = np.where(interacted_political)[0]
            other_idx = np.where(interacted_other)[0]

            political_pos = np.mean(self.parameters["content_representation"][political_idx])
            other_pos = np.mean(self.parameters["content_representation"][other_idx])
            self.parameters["position_matrix"][user, self.video_parameters["political"] == 1] = political_pos
            self.parameters["position_matrix"][user, self.video_parameters["political"] == 0] = other_pos

    def get_similarity_matrix(self):

        similarity_matrix = np.abs(self.parameters["position_matrix"]
                                   - self.parameters["content_representation"])
        # important step here, avoids that items get recommended twice
        # similarity_matrix = similarity_matrix - self.parameters["interaction_matrix"]
        self.parameters["similarity_matrix"] = np.ones((self.n_agents, self.n_videos)) - similarity_matrix


    def make_recommendation(self):
        self.parameters["current_recommendation"] = np.zeros_like(self.parameters["current_recommendation"])
        #rec_score = np.dot(self.parameters["similarity_matrix"], self.get_popularity())
        rec_score = self.parameters["similarity_matrix"] * self.get_popularity() #push this to self
        recommendations = np.argsort(rec_score, axis=1)[:, -100:] #TODO: make this a variable
        for user in range(self.n_agents):
            self.parameters["current_recommendation"][user, recommendations[user, :]] = 1

    def get_rec_score(self, alpha=0.5):
        self.parameters["rec_score"] = self.parameters["similarity_matrix"] * (self.get_popularity() **alpha)

    def rec_step(self):
        # position the user into the content space
        self.get_user_position()
        # get similarity of user position with content
        self.get_similarity_matrix()
        # make recommendation based on user similarity and overall popularity
        #self.make_recommendation()
        self.get_rec_score()

    def interaction_probability(self):
        ones = np.ones((self.n_agents, self.n_videos))
        dif = np.abs(self.user_parameters["opinion"].reshape(-1,1) - self.video_parameters["opinion"].reshape(1, -1))

        interaction_prob = self.video_parameters["gamma"] * (ones - dif)
        #self.parameters["interaction_probability"] = self.parameters["current_recommendation"] * interaction_prob
        self.parameters["interaction_probability"] = interaction_prob

    def n_interactions(self, theta=0.5, first_step=False):
        """
        A function that solves two problems: All users have to interact with the same videos and
        interactions happen probabilistic and not above a certain threshold.
        :return:
        """
        interaction_matrix = np.zeros((self.n_agents, self.n_videos))
        self.interaction_probability()
        for user in range(self.n_agents):
            if first_step:
                recommended_videos = np.arange(self.n_videos)
                np.random.shuffle(recommended_videos)
            else:
                recommended_videos = np.argsort(-self.parameters["rec_score"][user, :])

            interacted_idx = []

            for video in recommended_videos:
                # get interaction probability of specific video
                proba = self.parameters["interaction_probability"][user, video]
                # probabilistic part
                choice = np.random.choice([1, 0], p=[proba, 1-proba])
                if choice == 1:
                    interacted_idx.append(video)
                if len(interacted_idx) == 5:
                    break

            interaction_matrix[user, interacted_idx] = 1

        self.parameters["interactions"] = interaction_matrix
        # introduce discounting factor for older interactions
        self.parameters["all_interactions"] = self.parameters["all_interactions"] * theta
        self.parameters["all_interactions"] += interaction_matrix
        self.parameters["true_interactions"] += interaction_matrix




    def video_interactions(self):
        thresh = 0.3 #this should be probabilistic..
        self.interaction_probability()
        interact = np.where(self.parameters["interaction_probability"] > thresh, 1, 0)
        #also here interact should yield strictly n amounts of videos for each user
        #this will potentially need a loop
        self.parameters["interactions"] = interact
        self.parameters["all_interactions"] += interact

    def user_step(self, theta=0.5, first_step=False):
        #self.video_interactions()
        self.n_interactions(theta=theta, first_step=first_step)
        self.recommended_opinions()
        self.opinion_update()
        #how to keep track of opinions

    def recommended_opinions(self):
        """
        function to get the mean of the considered opinions
        :return:
        """

        #make the recommendation matrix row_stochastic, but only consider political opinions?
        political_considerations = self.parameters["interactions"] * self.video_parameters["political"] #maybe wrong
        result = np.dot((political_considerations / political_considerations.sum(axis=1).reshape(-1, 1)), self.video_parameters["opinion"])
        #this should yield a vector with shape (n_agents, 1), which would be then the opinion at
        # will result in na's if not considered for
        if np.any(np.isnan(result)):
            na_idx = np.where(np.isnan(result))
            result[na_idx] = self.user_parameters["opinion"][na_idx]

        self.user_parameters["considered_opinions"] = result

    def run_simulation(self, t=50):
        opinions = [self.user_parameters["opinion"]]
        self.initialize_recsys()
        self.user_step()
        opinions.append(self.user_parameters["opinion"])
        for _ in range(t):
            self.rec_step()
            self.user_step()
            opinions.append(self.user_parameters["opinion"])

        return opinions
    def better_simulation(self, t=50):
        self.parameters["all_interactions"] = np.zeros((self.n_agents, self.n_videos))
        self.parameters["position_matrix"] = np.ones((self.n_agents, self.n_videos))
        self.parameters["true_interactions"] = np.zeros((self.n_agents, self.n_videos))
        opinions = [self.user_parameters["opinion"]]
        self.user_step(first_step=True)
        opinions.append(self.user_parameters["opinion"])
        for _ in range(t):
            self.rec_step()
            self.user_step(first_step=False)
            opinions.append(self.user_parameters["opinion"])

        return opinions

    def get_attributes(self):
        return self.parameters["true_interactions"] #hmm