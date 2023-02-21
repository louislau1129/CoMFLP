import os
import pdb
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="clip12to6_ft_simple", required=True,
                    help="the output_dir of the asr saved dir.") 
    parser.add_argument("--mode", type=str, default="valid", 
                    help="you can choose average last num (last) \
                        or the lowest valid loss num (valid)") 
    parser.add_argument("--num", type=int, default=10)

    parser.add_argument("--last_epoch", type=int, default=50, required=True)
    args = parser.parse_args()


    exp_dir = os.path.join('exp', args.exp_dir)
    assert os.path.exists(exp_dir), f"{exp_dir} does not exist."

    avg_model_name = os.path.join(exp_dir, f"asr_avg_{args.mode}{args.num}.pt")
    


    if args.mode == "last":
        start_epoch = args.last_epoch - args.num + 1
        model_list = [
            os.path.join(exp_dir, f"asr_ep{epoch}.pt")
            for epoch in range(start_epoch, args.last_epoch+1)
        ]
    
    if args.mode == "valid":
        log_file = os.path.join(exp_dir, "train.log")
        epoch2wer = {}
        with open(log_file, 'r') as fd:
            for line in fd:
                if "Test wer :" in line:
                    wer = line.strip().split("Test wer :")[1]
                    wer = float(wer.strip().split("%")[0])
                    epoch = line.strip().split("EPOCH:")[1]
                    epoch = int(epoch.split(", Test wer")[0].strip())
                    epoch2wer[epoch] = wer
        
        sorted_epoch2wer = sorted(epoch2wer.items(), key=lambda kv: kv[1])
        epochs = [k[0] for k in sorted_epoch2wer]
        epochs = epochs[:args.num]
        model_list = [
            os.path.join(exp_dir, f"asr_ep{epoch}.pt")
            for epoch in epochs
        ]
    

    avg = None
   
    # sum
    for path in model_list:
        states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= args.num
            else:
                avg[k] //= args.num

    torch.save(avg, avg_model_name) 
    print(f"{avg_model_name} saved done")




    
    pass