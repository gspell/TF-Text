import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

""" From train_autoencoder.py, csv files are printed with columns as follows:
train_acc, train_loss, train_auc, eval_acc, eval_loss, eval_auc
"""

""" The type of postprocessing depends on what sort of runs were conducted: cross-val or varying hyperparameters"""

num_runs =8
total_runs = [r+1 for r in range(num_runs)]

def tuning_loss_ratio():
  averages = []
  file_list = ["tune_imdb_0.csv", "tune_imdb_0.001.csv", "tune_imdb_0.01.csv", "tune_imdb_0.1.csv", "tune_imdb_0.5.csv", 
               "tune_imdb_1.csv", "tune_imdb_5.csv", "tune_imdb_10.csv", "tune_imdb_100.csv", "tune_imdb_1000.csv", 
               "tune_imdb_10000.csv", "tune_imdb_100000.csv", "tune_imdb_1000000.csv"]
  for filename in file_list:
    averages.append(avg_auc(filename))
  return averages

def avg_auc(filename):
  df = pd.read_csv(filename, header=None)
  avg = np.mean(df.values)
  return avg

def extract_columns(run_num, directory):
  filename_auc = directory + "run"+ str(run_num) + "auc.csv"
  #filename_loss = directory + "run" + str(run_num) + "losses.csv"
  df_auc = pd.read_csv(filename_auc, header=None)
  acc = df_auc.values[:, 0::2]
  auc = df_auc.values[:, 1::2]
  #df_loss = pd.read_csv(filename_loss, header=None)
  #total_loss = df_loss.values[:, 0::3]
  #class_loss = df_loss.values[:, 1::3]
  #recon_loss = df_loss.values[:, 2::3]
  #return (acc, auc, total_loss, class_loss, recon_loss)
  return (acc, auc)

def plot_quantity(quantity_name, quantity, color):
  num_points = len(quantity)
  plt.plot(xrange(num_points), quantity[:,0], color, label="Train")
  plt.plot(xrange(num_points), quantity[:,1], color+'--', label="Eval")
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel(quantity_name)
  return plt

def plot_losses(losses):
  num_points = np.shape(losses[0])[0]
  plt.plot(xrange(num_points), losses[0][:,0], 'b', label="Train Total")
  plt.plot(xrange(num_points), losses[0][:,1], 'b--', label="Eval Total")
  plt.plot(xrange(num_points), losses[1][:,0], 'r', label="Train Class")
  plt.plot(xrange(num_points), losses[1][:,1], 'r--', label="Eval Class")
  plt.plot(xrange(num_points), losses[2][:,0], 'k', label="Train Recon")
  plt.plot(xrange(num_points), losses[2][:,1], 'k--', label="Eval Recon")
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Losses")
  return plt

def plot_results(plot_directory, results, run_num):
  num_points = np.shape(results)[1]
  acc_filename = plot_directory + "/run" + str(run_num) + "acc.png"
  acc = results[0]
  plt = plot_quantity("Accuracy", acc, 'b')
  plt.savefig(acc_filename)
  plt.close()
  auc_filename = plot_directory + "/run" + str(run_num) + "auc.png"
  auc = results[1]
  plt = plot_quantity("AUC", auc, 'b')
  plt.savefig(auc_filename)
  plt.close()
  """
  loss_filename = plot_directory + "/run" + str(run_num) + "loss.png"
  losses = results[2:5]
  plt = plot_losses(losses)
  plt.savefig(loss_filename)
  plt.close()
  """
def get_run_auc(results):
  eval_auc = results[1][:,1]
  last_vals = eval_auc[-10:]
  return np.mean(last_vals)

def get_run_loss(results):
  eval_loss = results[2][:,1]
  last_vals = eval_loss[-10:]
  return np.mean(last_vals)

def process_run(run_num, directory):
  run_results = extract_columns(run_num, directory)
  # plot_results(directory, run_results, run_num)
  auc_mean = get_run_auc(run_results)
  loss_mean = get_run_loss(run_results)
  return auc_mean, loss_mean

def get_summarized_quantities(directory):
  aucs, losses = [], []
  for run in total_runs:
    auc, loss = process_run(run, directory)
    aucs.append(auc)
    losses.append(loss)
  return aucs, losses

def curves_0_1():
  ratio2_0_5_1_to_9 = "./ratio.1/ratio2_.5_1_to_9/"
  ratio2_0_75_1_to_9 = "./ratio.1/ratio2_.75_1_to_9/"
  
  aucs_1, _ = get_summarized_quantities(ratio2_0_5_1_to_9)
  aucs_2, _ = get_summarized_quantities(ratio2_0_75_1_to_9)
  
  percentages = [run * 0.03275862 for run in total_runs]
  plt.figure()
  plt.plot(percentages, aucs_1, 'b', label="Ratio2 =  0.5")
  plt.plot(percentages, aucs_2, 'r', label="Ratio2 = 0.75")
  plt.legend(loc=0)
  plt.title("Ratio 1 = 0.1")
  plt.xlabel("Percentage of labeled training set")
  plt.ylabel("AUC on Eval Set")
  plt.savefig("tuning_ratio_2.png")
  plt.close()

def compare_MLP():
  linear = "./ratio.1/more_unlabeled/"
  MLP = "../realMLP/"
  
  aucs_1, _ = get_summarized_quantities(linear)
  aucs_2, _ = get_summarized_quantities(MLP)
  
  percentages = [run * 0.05 for run in total_runs]
  plt.figure()
  plt.plot(percentages, aucs_1, 'b', label="Linear")
  plt.plot(percentages, aucs_2, 'r', label="MLP")
  plt.legend(loc=0)
  plt.title("Ratio 1 = 0.1")
  plt.xlabel("Percentage of labeled training set")
  plt.ylabel("AUC on Eval Set")
  plt.savefig("mlp_comparison.png")
  plt.close()
def ratio_0_1():
  super_0_1_redone2 = "./ratio.1/supervised_new_batching2/"
  semi_0_1_many = "./ratio.1/more_unlabeled/"
  
  aucs_1, _ = get_summarized_quantities(super_0_1_redone2)
  aucs_2, _ = get_summarized_quantities(semi_0_1_many)
  
  percentages = [run * 0.05 for run in total_runs]
  plt.figure()
  plt.plot(percentages, aucs_1, 'b', label="Supervised")
  plt.plot(percentages, aucs_2, 'r', label="SemiSupervised")
  plt.legend(loc=0)
  plt.title("Ratio 1 = 0.1")
  plt.xlabel("Percentage of labeled training set")
  plt.ylabel("AUC on Eval Set")
  plt.savefig("ratio.1_auc.png")
  plt.close()
def create_yunchen_curves():
  # For loss ratio 1 = 0
  super_0 = "./ratio0/cell128_emb256_supervised/"
  
  # For loss ratio 1 = 0.001
  super_001 = "./ratio.001/cell128_emb256_supervised/"
  
  # For loss ratio 1 = 0.1
  semi_0_1_pretrain = "./ratio.1/cell128_emb256_semi_pretrain5000/"
  semi_0_1_10 = "./ratio.1/cell128_emb256_semi_10_unlabeled/"
  semi_0_1_2 = "./ratio.1/cell128_emb256_semi_2_unlabeled/"
  super_0_1 = "./ratio.1/cell128_emb256_supervised/"
  double_check = "./ratio.1/double_check_10/"
  new_batching = "./ratio.1/new_batching/"
  new_batching1_9 = "./ratio.1/new_batching_1_to_9/"
  new_batching2 = "./ratio.1/ratio2_.2/"
  super_0_1_redone = "./ratio.1/supervised_new_batching/"
  super_0_1_redone2 = "./ratio.1/supervised_new_batching2/"

  # For loss ratio 1 = 0.05
  semi_05 = "./ratio.05/cell128_emb256_semi_ratio2.1_nopretrain/"
  super_05 = "./ratio.05/cell128_emb256_supervised/"
  
  # For loss ratio 1 = 0.01
  semi_01 = "./ratio.01_nopretrain/cell128_emb256_semi/"
  super_01 = "./ratio.01_nopretrain/cell128_emb256_supervised/"
  
  # For loss ratio 1 = 1
  super1 = "./ratio1_nopretrain/"
  
  # For loss ratio 1 = 0.5
  semi_0_5_pretrain = "./ratio.5/cell128_emb256_semi_pretrain5000/"
  
  # The supervised runs
  aucs_01_super, losses_01_super = get_summarized_quantities(super_01)
  aucs_0_1_super, losses_0_1_super = get_summarized_quantities(super_0_1)
  aucs1_super, losses1_super = get_summarized_quantities(super1)
  aucs_05_super, losses_05_super = get_summarized_quantities(super_05)
  aucs_0_super, losses_0_super = get_summarized_quantities(super_0)
  #aucs_001_super, losses_001_super = get_summarized_quantities(super_001)
  #aucs_0_1_super_new, losses_0_1_super_new = get_summarized_quantities(super_0_1_redone)
  aucs_0_1_super_new2, losses_0_1_super_new = get_summarized_quantities(super_0_1_redone2)
  
  # Pretrained runs
  aucs_0_1_pretrain, losses_0_1_pretrain = get_summarized_quantities(semi_0_1_pretrain)
  aucs_0_5_pretrain, losses_0_5_pretrain = get_summarized_quantities(semi_0_5_pretrain)
  
  # Semisupervised runs
  aucs_0_1_semi2, losses_0_1_semi2 = get_summarized_quantities(semi_0_1_2)
  aucs_05_semi, losses_05_semi = get_summarized_quantities(semi_05)
  aucs_01_semi, losses_01_semi = get_summarized_quantities(semi_01)
  aucs_0_1_semi10, losses_0_1_semi10 = get_summarized_quantities(semi_0_1_10)
  aucs_double_check, losses_double_check = get_summarized_quantities(double_check)
  aucs_new_batching, losses_new_batching = get_summarized_quantities(new_batching)
  aucs_new_batching1_9, losses_new_batching1_9 = get_summarized_quantities(new_batching1_9)
  #aucs_new_batching2, losses_new_batching2 = get_summarized_quantities(new_batching2)
  
  percentages = [run * 0.05 for run in total_runs]
  plt.figure(0)
  
  plt.figure()
  plt.plot(percentages, aucs_0_1_pretrain, 'b', label="Ratio 0.1")
  plt.plot(percentages, aucs_0_5_pretrain, 'r', label="Ratio 0.5")
  plt.legend(loc=0)
  plt.title("Using same Pretraining - No semisupervision")
  plt.xlabel("Percentage of labeled training set")
  plt.ylabel("AUC on Eval Set")
  plt.savefig("pretraining_comparison.png")
  plt.close()
  
  plt.figure()
  #plt.plot(percentages, aucs_0_1_super_new, 'r--', label="Supervised Redone")
  plt.plot(percentages, aucs_0_1_super_new2, 'r', label="Supervised Redone 2")
  plt.plot(percentages, aucs_0_1_super, 'b', label="Supervised")
  #plt.plot(percentages, aucs_0_1_pretrain, 'r', label="Pretrained")
  #plt.plot(percentages, aucs_0_1_semi10, 'g', label="10 Unlabeled Batches")
  #plt.plot(percentages, aucs_0_1_semi2, 'k', label="2 Unlabeled Batches")
  #plt.plot(percentages, aucs_double_check, 'm', label="Double Check 10")
  plt.plot(percentages, aucs_new_batching, 'y', label="New Batching 4 to 6")
  plt.plot(percentages, aucs_new_batching1_9, 'g', label="New Batching 1 to 9")
  #plt.plot(percentages, aucs_new_batching2, 'k', label="Double Loss Ratio 2")
  plt.legend(loc=0)
  plt.title("Loss Ratio 1: 0.1")
  plt.xlabel("Percentage of labeled training set")
  plt.ylabel("AUC on Eval Set")
  plt.savefig("ratio.1_auc.png")
  plt.close()
  
  plt.figure()
  plt.plot(percentages, aucs_01_super, 'b', label="0.01")
  plt.plot(percentages, aucs_0_1_super, 'r', label="0.1")
  plt.plot(percentages, aucs1_super, 'g', label="1")
  plt.plot(percentages, aucs_05_super, 'k', label="0.05")
  plt.plot(percentages, aucs_0_super, 'm', label="0")
  #plt.plot(percentages, aucs_001_super, 'y', label="0.001")
  plt.legend(loc=0)
  plt.title("Supervised Varying Loss Ratios")
  plt.xlabel("Percentage of labeled training set")
  plt.ylabel("AUC on Eval Set")
  plt.savefig("ratio_variation_auc.png")
  plt.close()
  
  """
  plt.figure()
  plt.plot(percentages, supervised_losses, 'b', label="Supervised")
  plt.plot(percentages, semi_losses_pre, 'r', label="Pretrained")
  plt.plot(percentages, semi_losses_10, 'g', label="10 Unlabeled Batches")
  plt.plot(percentages, semi_losses_2, 'k', label="2 Unlabeled Batches")
  plt.legend(loc=0)
  plt.title("Both Loss Ratios: 0.1")
  plt.xlabel("Percentage of labeled training set")
  plt.ylabel("Loss on Eval Set")
  plt.savefig("ratio.1_loss.png")
  """
