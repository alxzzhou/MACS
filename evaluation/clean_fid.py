import argparse

from cleanfid import fid as F

ps = argparse.ArgumentParser()
ps.add_argument("-d", type=str)
ps.add_argument('-m', type=str)
ps.add_argument('-p', nargs=2, type=str)
args = ps.parse_args()
dataset, model = args.d, args.m

path1, path2 = args.p

fid_score = F.compute_fid(path1, path2)
kid_score = F.compute_kid(path1, path2)
clip_fid = F.compute_fid(path1, path2, mode='clean', model_name='clip_vit_b_32')

print(f'FID Score is: {fid_score:.2f}')
print(f'KID Score is: {kid_score:.4f}')
print(f'CLIP-FID Score is: {clip_fid:.2f}')
