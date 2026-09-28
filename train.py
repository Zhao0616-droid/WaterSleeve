from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    model = EDGE(
        opt.feature_type,
        checkpoint_path=opt.checkpoint,
        num_styles=opt.num_styles,
        attr_list=[a for a in opt.attr_list.split(",") if a],
        style_drop_prob=opt.style_drop_prob,
        attr_drop_prob=opt.attr_drop_prob,
        wrist_smooth_weight=opt.wrist_smooth_weight,
        arc_weight=opt.arc_weight,
        penetration_weight=opt.penetration_weight,
        penetration_thresh=opt.penetration_thresh,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
