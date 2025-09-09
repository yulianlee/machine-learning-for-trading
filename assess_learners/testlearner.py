""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
"""  		  	   		 	 	 			  		 			     			  	 
import math  		  	   		 	 	 			  		 			     			  	 
import sys  		  	   		 	 	 			  		 			     			  	  	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
import DTLearner as dt	     			  	 
import RTLearner as rt	  	  
import BagLearner as bl 	
import matplotlib.pyplot as plt	 	 	 			  		 			     			  	 
from datetime import datetime


def plot_figure(fig_num, train_rmses, test_rmses, title, labels, x_axis_label = "Leaf Size", y_axis_label = "RMSE Loss"):
    plt.figure()
    plt.gca().invert_xaxis()
    leaf_sizes = sorted(train_rmses.keys(), reverse=True)
    train_values = [train_rmses[ls] for ls in leaf_sizes]
    test_values = [test_rmses[ls] for ls in leaf_sizes]
    plt.plot(leaf_sizes, train_values, label=labels[0])
    plt.plot(leaf_sizes, test_values, label=labels[1])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.savefig(f"images/figure_{fig_num}.png", 
                bbox_inches='tight', 
                dpi=300)
    plt.close()   		

def calculate_r_squared(predictions, y_true):
    ss_res = np.sum((y_true - predictions)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return (1 - ss_res/ss_tot)


if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		 	 	 			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 			  		 			     			  	 
        sys.exit(1)  		  	   		 	 	 			  		 			     			  	 
    inf = open(sys.argv[1])  		 
    header = inf.readline() 	   		 	 	 			  		 			     			  	 
    data = np.array(  		  	   		 	 	 			  		 			     			  	 
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()]  		  	   		 	 	 			  		 			     			  	 
    )  		  	   		 	

    np.random.seed(42) # Set random seed

    # compute how much of the data is training and testing  	
    train_index = np.random.choice(len(data), size = round( 0.6 * len(data)), replace = False)	  	
    test_index = np.array([i for i in np.arange(len(data)) if i not in train_index])  		 	 	 			  		 			     			  	 
    train_rows = data[train_index]	  
    test_rows = data[test_index]	   

    # separate out training and testing data  			  	
    train_x = train_rows[:, :-1]
    train_y = train_rows[:, -1]
    test_x = test_rows[:, :-1]
    test_y = test_rows[:, -1]    
  		  	   		 	 	 			  		 			     			  	 	   		 	 	 			  		 			     			  	 	 	 	 			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		 	 	 			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		 	 	 			  		 			     			  	 
    
    # EXPERIMENT 1
    train_rmses = {}
    test_rmses = {}

    for leaf_size in np.arange(50, 1, -5):  	   		 	  		 			     			  	 	 
        dt_learner = dt.DTLearner(leaf_size = leaf_size) 	 			  		 			     			  	 
        dt_learner.add_evidence(train_x, train_y)

        # train RMSE
        dt_pred_y = dt_learner.query(train_x)
        dt_rmse_train = math.sqrt(((train_y - dt_pred_y) ** 2).sum() / train_y.shape[0])
        train_rmses[leaf_size] = dt_rmse_train

        # test RMSE
        dt_pred_y_test = dt_learner.query(test_x)
        dt_rmse_test = math.sqrt(((test_y - dt_pred_y_test) ** 2).sum() / test_y.shape[0])
        test_rmses[leaf_size] = dt_rmse_test
    
    plot_figure(1, train_rmses, test_rmses, "Train vs Test RMSE Loss", ['Train', 'Test'])		
    
    # EXPERIMENT 2
    train_rmses = {}
    test_rmses = {}

    for leaf_size in np.arange(50, 1, -5):  	   		 	  		 			     			  	 	 
        bag_learner = bl.BagLearner(dt.DTLearner, bags = 20, kwargs={"leaf_size":leaf_size})			  		 			     			  	 
        bag_learner.add_evidence(train_x, train_y)
        # train RMSE
        bag_pred_y = bag_learner.query(train_x)
        bag_rmse_train = math.sqrt(((train_y - bag_pred_y) ** 2).sum() / train_y.shape[0])
        train_rmses[leaf_size] = bag_rmse_train

        # test RMSE
        bag_pred_y_test = bag_learner.query(test_x)
        bag_rmse_test = math.sqrt(((test_y - bag_pred_y_test) ** 2).sum() / test_y.shape[0])
        test_rmses[leaf_size] = bag_rmse_test

    plot_figure(2, train_rmses, test_rmses, "BagLearner (Bag Size of 20) Train vs Test RMSE Loss",['Train', 'Test'])

    # EXPERIMENT 3
    time_dt = {}
    time_rt = {}
    r_squared_dt = {}
    r_squared_rt = {}

    for leaf_size in np.arange(50, 1, -5):  	
        dt_time_now = datetime.now()
        dt_learner = dt.DTLearner(leaf_size = leaf_size) 	 			  		 			     			  	 
        dt_learner.add_evidence(train_x, train_y)
        dt_pred_y = dt_learner.query(test_x)
        time_dt[leaf_size] = (datetime.now() - dt_time_now).total_seconds()
        r_squared_dt[leaf_size] = calculate_r_squared(dt_pred_y, test_y)

        rt_time_now = datetime.now()
        rt_learner = rt.RTLearner(leaf_size=leaf_size)
        rt_learner.add_evidence(train_x, train_y)
        rt_pred_y = rt_learner.query(test_x)
        time_rt[leaf_size] = (datetime.now() - rt_time_now).total_seconds()
        r_squared_rt[leaf_size] = calculate_r_squared(rt_pred_y, test_y)

    plot_figure(3, 
                time_dt, 
                time_rt, 
                "Total time taken for training and inference DTLearner vs RTLearner", 
                ['DTLearner', 'RTLearner'],
                x_axis_label = "Leaf Size",
                y_axis_label = "Time (s)"
                )
    plot_figure(4, 
                r_squared_dt, 
                r_squared_rt, 
                "R-Squared of DTLearner vs RTLearner", 
                ['DTLearner', 'RTLearner'],
                x_axis_label = "Leaf Size",
                y_axis_label = "Time (s)"
                )
