_base_ = './cfg_udd5.py'

model = dict(
    name_path='./configs/cls_demo9.txt',
    prob_thd=0.1,
    bg_idx=0,
    cls_token_lambda=-0.3,
)
