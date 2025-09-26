import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader

from codes.methods.networks.pmam.attn import Standard, PMAMWithBERT
from codes.methods.networks.seq2seq import (
    Seq2Seq,
    SinoNomDataset,
    collate_fn,
)
from codes.methods.networks.pmam.losses import DecodingLoss, MCGLoss
from codes.methods.utils import (
    greedy_decoding,
    pad_y_shift_to_string,
    compute_avg_cer,
    SOS_ID,
    PAD_ID,
    EOS_ID,
    UNK_ID,
)


def run_dry_run(
    model, train_dl, val_dl, device, id_to_token, decoding_loss_fn, mcg_loss_fn
):
    """
    Perform a dry run to test the forward pass of the model.
    """
    print("=== DRY RUN MODE ===")
    print("Testing forward pass of the model...")
    model.eval()

    # Test with one batch from training data
    for pad_x, pad_y_shift, pad_y, pad_b, viet_texts, mask in train_dl:
        pad_x = pad_x.to(device)
        pad_y_shift = pad_y_shift.to(device)
        pad_y = pad_y.to(device)
        pad_b = pad_b.to(device)
        mask = mask.to(device)

        print(f"Input batch shape: {pad_x.shape}")
        print(f"Target batch shape: {pad_y.shape}")
        print(f"Mask batch shape: {mask.shape}")
        print(f"Number of Vietnamese texts: {len(viet_texts)}")

        with torch.no_grad():
            try:
                # Test training mode forward pass
                print("\nTesting training mode forward pass...")
                logits, l = model(pad_x, viet_texts, pad_y_shift, mask)
                print(f"Training mode - Logits shape: {logits.shape}")
                if l is not None:
                    print(f"Training mode - Loss component shape: {l.shape}")
                else:
                    print("Training mode - No additional loss component")

                # Test inference mode forward pass
                print("\nTesting inference mode forward pass...")
                logits_inf, l_inf = model(pad_x, viet_texts, None, mask)
                print(f"Inference mode - Logits shape: {logits_inf.shape}")
                if l_inf is not None:
                    print(f"Inference mode - Loss component shape: {l_inf.shape}")
                else:
                    print("Inference mode - No additional loss component")

                # Test loss calculation
                print("\nTesting loss calculation...")
                loss = decoding_loss_fn(logits, pad_y, l, pad_b, mask)
                print(f"Decoding loss: {loss.item():.4f}")

                if l is not None:
                    mcg_loss = mcg_loss_fn(logits, pad_y, l, pad_b, mask)
                    total_loss = loss + mcg_loss
                    print(f"MCG loss: {mcg_loss.item():.4f}")
                    print(f"Total loss: {total_loss.item():.4f}")

                # Test CER computation
                print("\nTesting CER computation...")
                pad_y_shift_cpu = pad_y_shift.to("cpu")
                predictions, gt = greedy_decoding(logits, mask, id_to_token), pad_y_shift_to_string(pad_y_shift_cpu, mask, id_to_token),
                cer = compute_avg_cer(predictions, gt)
                print(f"CER: {cer:.4f}")

                print("\n=== DRY RUN SUCCESSFUL ===")
                print("Forward pass test completed successfully!")
                break

            except Exception as e:
                print("\n=== DRY RUN FAILED ===")
                print(f"Error during forward pass: {str(e)}")
                import traceback

                traceback.print_exc()
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bypass_check", action="store_true", help="Bypass 1-1 mapping check")
    parser.add_argument("--print_batch_step", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_res_path", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--transform", type=str, choices=["Standard", "PMAMWithBERT"], default="Standard"
    )
    parser.add_argument(
        "--phonetic", type=str, choices=["PhoBERTEncoder", "BARTphoEncoder", "None"], default="None"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a dry run to test the forward pass without training",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.input_dir, "vocab.txt")) as f:
        chars = [line.strip() for line in f if line.strip()]
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<sos>": SOS_ID, "<eos>": EOS_ID}
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    vocab_size = len(vocab)
    id_to_token = {v: k for k, v in vocab.items()}

    with open(os.path.join(args.input_dir, "train.txt")) as f:
        train_lines = f.readlines()
    with open(os.path.join(args.input_dir, "val.txt")) as f:
        val_lines = f.readlines()

    train_ds = SinoNomDataset(train_lines, vocab, args.bypass_check)
    val_ds = SinoNomDataset(val_lines, vocab, args.bypass_check)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    embed_dim = 256
    hidden_size = 384
    att_dim = 2 * hidden_size
    gate_hid = 256
    att_heads = 8

    transform_blk = args.transform
    phonetic_blk = args.phonetic
    if transform_blk != "Standard":
        from codes.methods.networks.pmam.encoders import PhoBERTEncoder, BARTphoEncoder

    model = Seq2Seq(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        att_dim=att_dim,
        drop_rate=0.3,
        transform=eval(transform_blk)(
            hidden_size=hidden_size,
            gate_hid=gate_hid,
            att_dim=att_dim,
            att_heads=att_heads,
            phonetic_enc=None if transform_blk == "Standard" else eval(phonetic_blk)(),
        ),
    )
    model.to(device)
    model.init_weights()
    if args.embeddings:
        model.load_pretrained_embeddings(args.embeddings)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # because we track CER
        factor=0.75,      # halve LR
        patience=1,      # wait 1 epoch
        min_lr=1e-4
    )
    # l2_lambda = 1e-5
    decoding_loss_fn = DecodingLoss(PAD_ID, scale_factor=1.0)
    mcg_loss_fn = MCGLoss(PAD_ID, scale_factor=args.scale)

    start_epoch = 0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    # If dry_run flag is set, run the dry run and exit
    if args.dry_run:
        run_dry_run(
            model, train_dl, val_dl, device, id_to_token, decoding_loss_fn, mcg_loss_fn
        )
        exit(0)

    logging.info(
        f"Loaded {len(train_ds)} training samples and {len(val_ds)} validation samples."
    )
    logging.info(f"Using device: {device}")

    print_batch_step = args.print_batch_step
    total_steps_per_epoch = len(train_dl)
    best_val_cer = float("inf")
    logging.info(f"Total steps per epoch: {total_steps_per_epoch}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        current_steps = 0
        for pad_x, pad_y_shift, pad_y, pad_b, viet_texts, mask in train_dl:
            current_steps += 1
            pad_x = pad_x.to(device)
            pad_y_shift = pad_y_shift.to(device)
            pad_y = pad_y.to(device)
            pad_b = pad_b.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            logits, l = model(pad_x, viet_texts, pad_y_shift, mask)
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = decoding_loss_fn(logits, pad_y, l, pad_b, mask)
            if l is not None:
                loss += mcg_loss_fn(logits, pad_y, l, pad_b, mask)
            # loss = loss + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()

            pad_y_shift = pad_y_shift.to("cpu")

            if current_steps % print_batch_step == 0:
                predictions, gt = greedy_decoding(logits, mask, id_to_token), pad_y_shift_to_string(pad_y_shift, mask, id_to_token),
                cer = compute_avg_cer(predictions, gt)
                logging.info(
                    f"Epoch {epoch}, Step {current_steps}/{total_steps_per_epoch}, Loss: {loss.item():.4f}, CER: {cer:.4f}"
                )

        logging.info(f"End of Epoch {epoch}")
        logging.info("Starting evaluation phase")

        model.eval()
        val_cer = 0.0
        with torch.no_grad():
            f = None
            if args.save_res_path is not None:
                f = open(args.save_res_path, "a")
                f.write(f"\n======EPOCH {epoch+1}/{args.epochs}======\n")

            for pad_x, pad_y_shift, pad_y, pad_b, viet_texts, mask in val_dl:
                pad_x = pad_x.to(device)
                pad_y_shift = pad_y_shift.to(device)
                pad_y = pad_y.to(device)
                pad_b = pad_b.to(device)
                mask = mask.to(device)
                logits, l = model(pad_x, viet_texts, pad_y_shift, mask)
                pad_y_shift = pad_y_shift.to("cpu")
                predictions, gt = greedy_decoding(logits, mask, id_to_token), pad_y_shift_to_string(pad_y_shift, mask, id_to_token),

                if f is not None:
                    for pred, truth in zip(predictions, gt):
                        f.write(f"Prediction: {pred}\tGround Truth: {truth}\n")

                cer = compute_avg_cer(predictions, gt) * len(viet_texts)
                val_cer += cer

            if f is not None:
                f.close()

        avg_val_cer = val_cer / len(val_ds)
        scheduler.step(avg_val_cer)
        
        if best_val_cer > avg_val_cer:
            best_val_cer = avg_val_cer
            best_checkpoint_path = os.path.join(
                args.output_dir, f"best_model.pt"
            )
            torch.save(model.state_dict(), best_checkpoint_path)
            logging.info(f"Saved checkpoint to {best_checkpoint_path}")
        
        logging.info(
            f"Epoch {epoch} Summary:\n"
            f"  Validation CER: {avg_val_cer:.4f}\n"
            f"  Total Validation Samples: {len(val_ds)}\n"
            f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        ckpt_path = os.path.join(
            args.output_dir, f"latest.pt"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            ckpt_path,
        )
