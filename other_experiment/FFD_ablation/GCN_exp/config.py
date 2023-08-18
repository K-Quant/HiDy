import  argparse

args = argparse.ArgumentParser()
args.add_argument('--model', default='gcn')
args.add_argument('--hidden', type=int, default=8)
args.add_argument('--dropout', type=float, default=0.6)
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--max_degree', type=int, default=3)



args = args.parse_args()
print(args)