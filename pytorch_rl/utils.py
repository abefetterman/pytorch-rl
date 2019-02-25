
def rewards_to_go(rewards, gamma=0.99):
    rev_rtg = []
    run_reward = 0
    for x in rewards[::-1]:
        run_reward = x + run_reward * gamma
        rev_rtg.append(run_reward)
    return rev_rtg[::-1]