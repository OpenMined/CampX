import os
import glob
import config
import argparse
import pandas as pd


def mp(args):
  # get current working directory children for analysis, ignore non-data directories
  directory = args.directory
  all_files = glob.glob(directory + "/*.csv")
  df = pd.concat((pd.read_csv(f) for f in all_files), axis=1)
  print(df)


if __name__ == "__main__":
  # first command line argument is the directory to process
  parser = argparse.ArgumentParser(description=('Plot group of'
                                                'experimental runs.'))
  parser.add_argument('-d', '--directory', default='output')
  parser.add_argument('-p', '--prefix', default='sarsa_nep_434')
  parser.add_argument('-m', '--plotMean', default='False', 
      help="Plot mean of groups, True or False")
  parser.add_argument('-x', '--figSizeX', default=12)
  parser.add_argument('-y', '--figSizeY', default=5)
  parser.add_argument('-b', '--buckets', default=config.REPORT_EVERY_N, 
      help="Average data into N buckets, use 0 for no averaging")
  parser.add_argument('-rt', '--rewardTitle', default='Average Return')
  parser.add_argument('-g', '--group', default='False', 
      help="Group by sweep parameter")
  parser.add_argument('-s', '--plotStd', default='False', 
      help="Plot standard deviation")
  args = parser.parse_args()
  mp(args)