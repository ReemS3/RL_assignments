# Review other groups' submission:
## Review of the first assigment:
  - Review the first assigment of group 37 [click here](https://docs.google.com/document/d/1zavPsYhRTxLldvB2dgnKQT_ywEHMgTMvbyESneLcljM/edit?usp=sharing).

  ## Review of group 11:

  - For tasks 2 and 3 I don't have anything to add or criticise, I believe our groups have mostly the same points written, but I like you're detailed answer and examples in the latter task. For the first task just a few remarks:
    - The assumption of a deterministic opponent is of course a rather strong one, this would mean that our problem rather is "play chess against this very specific and boring opponent" instead of "play chess against others".
    - There might be some problems with your reward dynamics: If an agent could do a check mate in one or three moves (or after capturing a pawn) it should just prefer the later check mate, which is not what a human would define as "best". Also there is an incentive to rather loose in ten moves than to do a remis (chess term for draw) next move. My suggestions would rather be to remove "artificial" rewards for capturing pieces or getting a check (which is most times not a real problem) but to include a negative reward for losing and small positive reward for draw.


  ## Review of the Group 38:

  ### Task 1

  - Great points are given! Through the context of the game of chess, one could also add a little more detail/analogy to explain whether the environment is stochastic or deterministic; explain the transition model and Markov property; be more specific about what a policy is, i.e. winning in chess could mean – “capture the opponent’s king and protect yours”. 

  ### Task 2

  - Nothing to add here, I think it looks very good.  

  ### Task 3

  - Good explanation of the reward and state transition functions. Perhaps, a brief introduction to the definitions of policy evaluation and optimal policy could be helpful as well. 
  - Looks good, but giving more specific examples for the cases with completely or partially known environment dynamics would be helpful. For example, one could start with the gridworld vs the real-world examples where, in the first case, the environment dynamics are completely known to the agent (complete observation=the agent can observe the complete state information that describes the environment/world; MDP framework), but only partially known in the real world example (partial observation=the agent can observe only partial information regarding the state of the environment). A suitable real-world analogy could be the self-driving car example mentioned in the lecture.
