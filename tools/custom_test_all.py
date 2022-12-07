import os
import mmcv
import argparse


def parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", type=str, required=True, help="config file relative path")
    parser.add_argument("--start", type=int, default=30000, help="start iter")
    parser.add_argument("--end", type=int, default=300000, help="end iter")
    parser.add_argument("--step", type=int, default=1000, help="step size")
    parser.add_argument("--byseq", action="store_true", default=False, help="by seq")
    return parser
    

if __name__ == "__main__":
    parser = parse()
    cfg = parser.parse_args()
    CONFIG = cfg.config
    CHECKPOINT = os.path.basename(os.path.splitext(CONFIG)[0])
    start = cfg.start
    end = cfg.end
    step = cfg.step
    total_iters = int(CHECKPOINT.rsplit('_',maxsplit=2)[1][:-1])*1000
    if cfg.end <= total_iters:
        end = cfg.end
    else:
        end = total_iters
    end += step
    iters = range(start, end, step)
    print(iters)
    if cfg.byseq:
        config = mmcv.Config.fromfile(CONFIG)
        seq = config.data.train.dataset.get("num_input_frames", 9)
        script = "tools/test_dehaze_video_inference.py"
        print(seq)
    else:
        script = "tools/custom_test.py"
    for iter in iters:
        cmd = f"""python {script} \
                            --config {CONFIG} \
                            --checkpoint work_dirs/{CHECKPOINT}/iter_{iter}.pth"""
        if cfg.byseq:
            cmd += f" --seq {seq}"
        print(cmd)
        os.system(cmd)
    
