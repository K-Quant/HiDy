import torch
from sklearn.metrics import f1_score
from utils import EarlyStopping, load_data
import numpy as np
import pandas as pd

def score(logits, labels):
    prediction = [0 if x < 0.5 else 1 for x in logits]
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, val_mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)

    loss = loss_func(logits[val_mask].squeeze().squeeze(), labels[val_mask].squeeze().squeeze().float())
    accuracy, micro_f1, macro_f1 = score(logits[val_mask].squeeze().squeeze(), labels[val_mask].squeeze().squeeze().float())

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data(args)


    features = features.to(args["device"])
    labels = labels.to(args["device"])
    train_mask = train_mask.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])

    from model import HAN
    model = HAN(
        num_meta_paths=len(g),
        in_size=features.shape[1],
        hidden_size=args["hidden_units"],
        out_size=num_classes,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
    ).to(args["device"])
    g = [graph.to(args["device"]) for graph in g]

    stopper = EarlyStopping(patience=args["patience"])

    pos_weight = torch.tensor([3.5])
    loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )


    for epoch in range(args["num_epochs"]):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask].squeeze().squeeze(), labels[train_mask].squeeze().squeeze().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_acc, train_micro_f1, train_macro_f1 = score(
        #     logits[train_mask].squeeze().squeeze(), labels[train_mask].squeeze().squeeze().float()
        # )
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        print(
            "Epoch {}, val loss {:.4f} | val Micro f1 {:.4f} | val Macro f1 {:.4f} | val accuracy {:.4f}".format(
                epoch, val_loss.item(), val_micro_f1, val_macro_f1, val_acc
            )
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    print(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test accuracy {:.4f}".format(
            test_loss.item(), test_micro_f1, test_macro_f1, test_acc
        )
    )


if __name__ == "__main__":
    import argparse
    from utils import setup

    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--Branch_lst", default=['IndustryChain','SectorIndustry','Ownership','Partnership'],
        help="Branch of HiDy for FFD tasks: IndustryChain, SectorIndustry, Ownership, Partnership",
    )
    args = parser.parse_args().__dict__
    args = setup(args)
    main(args)


