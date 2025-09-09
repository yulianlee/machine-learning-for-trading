""""""  		  	   		 	 	 			  		 			     			  	 
"""Assess a betting strategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Yulian Lee Ying Hern		  	   		 	 	 			  		 			     			  	 
GT User ID: yhern3 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 903870865 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
import matplotlib.pyplot as plt	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def author():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "yhern3"  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def gtid():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: int  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return 903870865	

def study_group():
    return "Yulian Lee Ying Hern"	  	   		 	 	 			  		 			     			  	 

def save_chart(figure_number):
    plt.savefig(f"images/figure_{figure_number}.png", 
                bbox_inches='tight', 
                dpi=300)
    plt.close()   			  	 
  		  	   		 	 	 			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			     			  	 
    :type win_prob: float  		  	   		 	 	 			  		 			     			  	 
    :return: The result of the spin.  		  	   		 	 	 			  		 			     			  	 
    :rtype: bool  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 		

    result = False  		  	   		 	 	 			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		 	 	 			  		 			     			  	 
        result = True  		  	   		 	 	 			  		 			     			  	 
    return result  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	

def one_episode(win_prob, max_spins=1000, target=80):
    """Return the win history for each spin in a single episode."""
    winnings = np.zeros(max_spins+1)
    episode_winnings = 0
    bet_amount = 1

    for spin in range(1, max_spins+1):
        if episode_winnings >= target:
            winnings[spin:] = episode_winnings
            break

        won = get_spin_result(win_prob)
        if won:
            episode_winnings += bet_amount
            bet_amount = 1
        else:
            episode_winnings -= bet_amount
            bet_amount *= 2

        winnings[spin] = episode_winnings

    return winnings[1:]

def one_episode_with_bankroll(win_prob, max_spins=1000, target=80):
    """Return the win history for each spin in a single episode with bankroll."""
    winnings = np.zeros(max_spins+1)
    episode_winnings = 0
    bet_amount = 1
    bankroll = 256

    for spin in range(1, max_spins+1):
        remaining_bankroll = bankroll + episode_winnings

        if bet_amount > remaining_bankroll:
            bet_amount = remaining_bankroll

        if episode_winnings >= target:
            winnings[spin:] = episode_winnings
            break

        won = get_spin_result(win_prob)
        if won:
            episode_winnings += bet_amount
            bet_amount = 1
        else:
            episode_winnings -= bet_amount
            bet_amount *= 2

        winnings[spin] = episode_winnings

        if bankroll + episode_winnings <= 0:
            winnings[spin:] = -256
            break

    return winnings[1:]




def plot_figure_1(win_prob):
    plt.figure(figsize=(10, 6))
    for _ in range(10):
        winnings = one_episode(win_prob)
        plt.plot(winnings, label="Episode")
    plt.axhline(y=0)
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.title("Figure 1: Winnings for 10 episodes")
    save_chart(1)

def plot_figure_2(win_prob):
    all_winnings = [one_episode(win_prob) for _  in range(1000)]
    all_winnings = np.array(all_winnings)
    all_winnings_mean = np.mean(all_winnings, axis = 0)
    all_winnings_std = np.std(all_winnings, axis = 0)

    plt.figure(figsize=(10, 6))
    plt.plot(all_winnings_mean, label = 'Mean winnings')
    plt.plot(all_winnings_mean + all_winnings_std, label = ('Mean winnings + Standard Deviation'))
    plt.plot(all_winnings_mean - all_winnings_std, label = ('Mean winnings - Standard Deviation'))
    plt.xlim(0,300)
    plt.ylim(-256, 100)
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.title("Figure 2: Mean winnings for 1000 episodes")
    plt.legend()
    save_chart(2)


def plot_figure_3(win_prob):
    all_winnings = np.array([one_episode(win_prob) for _  in range(1000)])
    all_winnings_median = np.median(all_winnings, axis = 0)
    all_winnings_std = np.std(all_winnings, axis = 0)

    plt.figure(figsize=(10, 6))
    plt.plot(all_winnings_median, label = 'Median winnings')
    plt.plot(all_winnings_median + all_winnings_std, label = ('Median winnings + Standard Deviation'))
    plt.plot(all_winnings_median - all_winnings_std, label = ('Median winnings - Standard Deviation'))
    plt.xlim(0,300)
    plt.ylim(-256, 100)
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.title("Figure 3: Median winnings for 1000 episodes")
    plt.legend()
    save_chart(3)


def plot_figure_4(win_prob):
    all_episode_winnings = np.array([one_episode_with_bankroll(win_prob) for _ in range(1000)])
    all_winnings_mean = np.mean(all_episode_winnings, axis = 0)
    all_winnings_std = np.std(all_episode_winnings, axis = 0)

    plt.figure(figsize=(10, 6))
    plt.plot(all_winnings_mean, label = 'Mean winnings with $256 bankroll')
    plt.plot(all_winnings_mean + all_winnings_std, label = 'Mean winnings with $256 bankroll + standard deviation')
    plt.plot(all_winnings_mean - all_winnings_std, label = 'Mean winnings with $256 bankroll - standard deviation')
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.title("Figure 4: Mean winnings with limited bankroll for 1000 episodes")
    plt.legend()
    plt.xlim(0,300)
    plt.ylim(-256, 100)
    save_chart(4)


def plot_figure_5(win_prob):
    all_episode_winnings = np.array([one_episode_with_bankroll(win_prob) for _ in range(1000)])
    all_winnings_median = np.median(all_episode_winnings, axis = 0)
    all_winnings_std = np.std(all_episode_winnings, axis = 0)

    plt.figure(figsize=(10, 6))
    plt.plot(all_winnings_median, label = 'Median winnings with $256 bankroll')
    plt.plot(all_winnings_median + all_winnings_std, label = 'Median winnings with $256 bankroll + standard deviation')
    plt.plot(all_winnings_median - all_winnings_std, label = 'Median winnings with $256 bankroll - standard deviation')
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.title("Figure 5: Median winnings with limited bankroll for 1000 episodes")
    plt.legend()
    plt.xlim(0,300)
    plt.ylim(-256, 100)
    save_chart(5)


def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Method to test your code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    win_prob = 18/38  # set appropriately to the probability of a win  		  	   		 	 	 			  		 			     			  	 
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 			  		 			     			  	 	  	   		 	 	 			  		 			     			  	 
    # add your code here to implement the experiments  	
    plot_figure_1(win_prob)
    plot_figure_2(win_prob)
    plot_figure_3(win_prob)
    plot_figure_4(win_prob)
    plot_figure_5(win_prob)

    # # Question Set 1
    # all_wins = [one_episode(win_prob) for _ in range(1000)]
    # total_times_reaching_80 = [i for i in all_wins if max(i) >= 80 ]
    # print(len(total_times_reaching_80))
    # # Question Set 2
    # all_wins = np.array(all_wins)
    # expected_value_one_episode = np.mean(all_wins[:, 999], axis = 0)
    # print(expected_value_one_episode)
    # # Question Set 4
    # all_wins_with_bankroll = [one_episode_with_bankroll(win_prob).tolist() for _ in range(1000)]
    # total_times_reaching_80_with_bankroll = [i for i in all_wins_with_bankroll if max(i) >= 80 ]  
    # from collections import Counter
    # print(Counter([i[999] for i in all_wins_with_bankroll]))
    # print(len(total_times_reaching_80_with_bankroll))
    # # Question Set 5
    # all_wins_with_bankroll = np.array(all_wins_with_bankroll)
    # expected_value_one_episode_with_bankroll = np.mean(all_wins_with_bankroll[:, 999], axis = 0)
    # print('exp', expected_value_one_episode_with_bankroll)
    # # counts, _, patches = plt.hist(all_wins_with_bankroll[:, 999], align='mid')
    # # plt.title('Figure 2: Cumulative winning amounts after 1000 sequential bets')
    # # plt.xlabel('Winning Amounts')
    # # plt.ylabel('Frequency')
    # # plt.xticks([-256, 80])
    # # for count, patch in zip(counts, patches):
    # #     plt.text(patch.get_x() + patch.get_width() / 2, count, int(count), 
    # #             ha='center', va='bottom')
    # # plt.show()

    # # Question Set 6
    # all_wins_with_bankroll_std_dev = np.std(all_wins_with_bankroll, axis = 0)
    # print(all_wins_with_bankroll_std_dev[-1])
    # print(np.mean(all_wins_with_bankroll, axis=0)[-1] + all_wins_with_bankroll_std_dev[-1])
    # print(np.mean(all_wins_with_bankroll, axis=0)[-1] - all_wins_with_bankroll_std_dev[-1])

if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
