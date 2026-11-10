# Library Imports
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
from PIL import Image, ImageDraw

from train_iql_offline import DATASET, choose_action, load_iql_policy, write_json

SEEDS = (0,1,2,3,4)

def read_json(path):
    return json.loads(path.read_text())

def save_rollout_gif(policy,dataset,seed,path):
    env=dataset.recover_environment(eval_env=True,render_mode="rgb_array",width=480,height=360)
    frames,episode_return,step=[],0.0,0
    try:
        obs,_=env.reset(seed=30000+seed)
        while True:
            if step%5==0:
                frame=Image.fromarray(env.render())
                draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,480,24),fill="white")
                draw.text((8,6),f"IQL seed {seed} | step {step} | return {episode_return:.1f}",fill="black")
                frames.append(frame)
            obs,reward,terminated,truncated,_=env.step(choose_action(policy,obs))
            episode_return+=float(reward)
            step+=1
            if terminated or truncated:
                frames.append(Image.fromarray(env.render()))
                break
        frames[0].save(path,save_all=True,append_images=frames[1:],duration=round(5000/env.metadata["render_fps"]),loop=0)
    finally:
        env.close()
    return {"training_seed":seed,"eval_seed":30000+seed,"return":episode_return,"length":step}

def main(args):
    torch.set_num_threads(1)
    runs=[args.results_dir/f"iql_seed{seed}" for seed in SEEDS]
    finals=[read_json(run/"final_eval.json") for run in runs]
    curves=[read_json(run/"learning_curve.json") for run in runs]
    if any(x["step"]!=1000000 or len(x["episodes"])!=20 for x in finals):
        raise ValueError("Five full-budget final policies with 20 evaluation episodes are required.")
    steps=list(range(50000,1000001,50000))
    if any([x["step"] for x in curve]!=steps or any(x["n_episodes"]!=10 for x in curve) for curve in curves):
        raise ValueError("Periodic evaluations must use ten episodes every 50,000 iterations.")
    values=np.array([[x["mean_return"] for x in curve] for curve in curves])
    mean,std=values.mean(0),values.std(0,ddof=1)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(steps,mean,label="IQL: mean over 5 training seeds")
    ax.fill_between(steps,mean-std,mean+std,alpha=.2,label="+/-1 sample std across seeds")
    ax.set(xlabel="Gradient iterations",ylabel="Average episodic return (raw)",title="Hopper medium-v0 | 10 evaluation episodes per seed")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.results_dir/"iql_learning_curve.png",dpi=300)
    plt.close(fig)
    rows=[]
    for seed,final in zip(SEEDS,finals):
        returns=np.array([x["return"] for x in final["episodes"]])
        rows.append({"seed":seed,"mean_return":float(returns.mean()),"episode_std":float(returns.std(ddof=1)),"n_episodes":20})
    with (args.results_dir/"iql_final_returns.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    returns=np.array([x["mean_return"] for x in rows])
    summary={"n_seeds":5,"mean_return":float(returns.mean()),"seed_std":float(returns.std(ddof=1)),"per_seed":rows}
    write_json(args.results_dir/"iql_summary.json",summary)
    if not args.skip_gifs:
        os.environ["MINARI_DATASETS_PATH"]=str(args.data_dir.resolve())
        dataset=minari.load_dataset(DATASET)
        rollouts=[]
        for seed in SEEDS:
            policy,saved=load_iql_policy(args.checkpoint_dir/f"iql_seed{seed}"/"final.pt")
            if saved["step"]!=1000000:
                raise ValueError("GIFs require the final full-budget checkpoints.")
            rollouts.append(save_rollout_gif(policy,dataset,seed,args.gif_dir/f"iql_seed{seed}.gif"))
        write_json(args.results_dir/"rollouts.json",rollouts)
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Generate IQL curves, return tables and five final-policy GIFs.")
    parser.add_argument("--results-dir",type=Path,default=Path("results"))
    parser.add_argument("--checkpoint-dir",type=Path,default=Path("checkpoints"))
    parser.add_argument("--data-dir",type=Path,default=Path("data"))
    parser.add_argument("--gif-dir",type=Path,default=Path("."))
    parser.add_argument("--skip-gifs",action="store_true")
    main(parser.parse_args())
