from typing import Tuple
import torch


@torch.no_grad()
def testing(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    output_filename: str,
    device: str = "cuda:0",
    amp: bool = True
) -> None:

    # A procedure for evaluating on the test dataset
    # There is no ground truth

    model.eval()

    output_fp = open(output_filename, "w")
    output_fp.write("newID, res\n")
    for idx, (inputs, newID) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        newID = newID[:, 0]

        with torch.autocast("cuda", enabled=amp):
            logits = model(inputs)
            # Aggregating logits accross time dimension, and then performing softmax. 
            pred = torch.softmax(logits.sum(1), dim=1)
            pred = torch.argmax(pred, dim=1)

        for sample_idx, _pred in enumerate(pred):
            output_fp.write(f"{int(newID[sample_idx])}, {int(_pred)}\n")

    output_fp.close()
    print(f"TESING DONE IN {output_filename}", flush=True)


@torch.no_grad()
def validate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str = "cuda:0",
    amp: bool = True
) -> Tuple[float, float]:

    # A procedure for evaluating on the val dataset
    # Perform to compute accuracy and loss

    model.eval()

    loss = 0.
    correct = 0
    accuracy = 0.
    sample_counter = 0.
    for idx, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]

        with torch.autocast("cuda", enabled=amp):
            logits = model(inputs)
            # logits.shape: (B, seq_length, 2)

            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ).sum()
            pred = torch.softmax(logits, dim=2)
            pred = torch.argmax(pred, dim=2)

            # one-hot to label
            targets = torch.argmax(targets, dim=2)
            correct += (pred == targets).sum()
            sample_counter += batch_size * seq_length

    accuracy = float(correct / sample_counter)
    loss = float(loss / sample_counter)

    return accuracy, loss
