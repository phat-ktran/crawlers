import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader

from codes.methods.networks.attentions import BahdanauAttention
from codes.methods.networks.decoders import Decoder
from codes.methods.networks.encoders import Encoder
from codes.methods.networks.encoders.bert_enc import PhoBERTEncoder
from codes.methods.networks.encoders.fuse import Fusion, Add, Concat, Sigmoid, Residual
from codes.methods.networks.encoders.sinonom_enc import SinoNomEncoder
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
    logging.debug("=== DRY RUN MODE ===")
    logging.debug("Testing forward pass of the model...")
    model.eval()

    # Test with one batch from training data
    for pad_x, pad_y_shift, pad_y, pad_b, viet_texts, mask in train_dl:
        pad_x = pad_x.to(device)
        pad_y_shift = pad_y_shift.to(device)
        pad_y = pad_y.to(device)
        pad_b = pad_b.to(device)
        mask = mask.to(device)

        logging.debug(f"Input batch shape: {pad_x.shape}")
        logging.debug(f"Target batch shape: {pad_y.shape}")
        logging.debug(f"Mask batch shape: {mask.shape}")
        logging.debug(f"Number of Vietnamese texts: {len(viet_texts)}")

        with torch.no_grad():
            try:
                # Test training mode forward pass
                logging.debug("\nTesting training mode forward pass...")
                logits, l = model(pad_x, viet_texts, pad_y_shift, mask)
                logging.debug(f"Training mode - Logits shape: {logits.shape}")
                if l is not None:
                    logging.debug(f"Training mode - Loss component shape: {l.shape}")
                else:
                    logging.debug("Training mode - No additional loss component")

                # Test inference mode forward pass
                logging.debug("\nTesting inference mode forward pass...")
                logits_inf, l_inf = model(pad_x, viet_texts, None, mask)
                logging.debug(f"Inference mode - Logits shape: {logits_inf.shape}")
                if l_inf is not None:
                    logging.debug(f"Inference mode - Loss component shape: {l_inf.shape}")
                else:
                    logging.debug("Inference mode - No additional loss component")

                # Test loss calculation
                logging.debug("\nTesting loss calculation...")
                loss = decoding_loss_fn(logits, pad_y, l, pad_b, mask)
                logging.debug(f"Decoding loss: {loss.item():.4f}")

                if l is not None:
                    mcg_loss = mcg_loss_fn(logits, pad_y, l, pad_b, mask)
                    total_loss = loss + mcg_loss
                    logging.debug(f"MCG loss: {mcg_loss.item():.4f}")
                    logging.debug(f"Total loss: {total_loss.item():.4f}")

                # Test CER computation
                logging.debug("\nTesting CER computation...")
                pad_y_shift_cpu = pad_y_shift.to("cpu")
                predictions, gt = (
                    greedy_decoding(logits, mask, id_to_token),
                    pad_y_shift_to_string(pad_y_shift_cpu, mask, id_to_token),
                )
                cer = compute_avg_cer(predictions, gt)
                logging.debug(f"CER: {cer:.4f}")

                logging.debug("\n=== DRY RUN SUCCESSFUL ===")
                logging.debug("Forward pass test completed successfully!")
                break

            except Exception as e:
                logging.debug("\n=== DRY RUN FAILED ===")
                logging.debug(f"Error during forward pass: {str(e)}")
                import traceback

                traceback.print_exc()
                break


def create_dataloaders(vocab, args):
    with open(os.path.join(args.input_dir, "train.txt")) as f:
        train_lines = f.readlines()
    with open(os.path.join(args.input_dir, "val.txt")) as f:
        val_lines = f.readlines()

    train_ds = SinoNomDataset(train_lines, vocab, args.bypass_check)
    val_ds = SinoNomDataset(val_lines, vocab, args.bypass_check)

    train_dl = DataLoader(
        train_ds,
        batch_size=4 if args.dry_run else args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=4 if args.dry_run else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return train_dl, val_dl, len(val_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument(
        "--emb_dims", type=int, default=256, help="Embedding dimensions for the model"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=384, help="Hidden size for the model"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--bypass_check", action="store_true", help="Bypass 1-1 mapping check"
    )
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--attn_drop_out", type=float, default=0.3, help="Dropout rate for attention layers")
    parser.add_argument("--drop_out", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("--print_batch_step", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_res_path", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["Fusion", "Concat", "Add", "Sigmoid", "Residual"],
        default="Fusion",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["Identity", "PhoBERTEncoder"],
        default="Identity",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a dry run to test the forward pass without training",
    )
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    if args.dry_run:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(message)s",
        )
    else:
        logging.basicConfig(
            filename=os.path.join(args.output_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    with open(os.path.join(args.input_dir, "vocab.txt")) as f:
        chars = [line.strip() for line in f if line.strip()]
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<sos>": SOS_ID, "<eos>": EOS_ID}
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    vocab_size = len(vocab)
    id_to_token = {v: k for k, v in vocab.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    embed_dim = args.emb_dims
    hidden_size = args.hidden_size
    ref_encoder = eval(args.encoder)()
    fuser = eval(args.fusion)(
        src_hidden_size=hidden_size,
        ref_hidden_size=ref_encoder.hidden_size,
        num_heads=args.heads,
        drop_out=args.attn_drop_out,
    )
    model = Seq2Seq(
        vocab_size=vocab_size,
        encoder=Encoder(
            src_enc=SinoNomEncoder(
                embed_dim=embed_dim,
                hidden_size=hidden_size,
            ),
            ref_enc=ref_encoder,
            fuser=fuser,
        ),
        attention=BahdanauAttention(
            enc_hidden_size=fuser.fused_dim,
            dec_hidden_size=hidden_size,
            att_dim=hidden_size,
        ),
        decoder=Decoder(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            context_dim=fuser.fused_dim,
            vocab_size=vocab_size,
        ),
        drop_rate=args.drop_out,
    )
    model.to(device)
    model.init_weights()
    if args.embeddings:
        model.load_pretrained_embeddings(args.embeddings)

    train_dl, val_dl, val_size = create_dataloaders(vocab, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # because we track CER
        factor=0.5,  # halve LR
        patience=1,  # wait 1 epoch
        min_lr=1e-4,
    )

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
            loss = decoding_loss_fn(logits, pad_y, l, pad_b, mask)
            if l is not None:
                loss += mcg_loss_fn(logits, pad_y, l, pad_b, mask)
            loss.backward()
            optimizer.step()

            pad_y_shift = pad_y_shift.to("cpu")
            if current_steps % print_batch_step == 0:
                predictions, gts = (
                    greedy_decoding(logits, mask, id_to_token),
                    pad_y_shift_to_string(pad_y_shift, mask, id_to_token),
                )
                predictions = [pred.replace("<unk>", "") for pred in predictions]
                gts = [gt.replace("<unk>", "") for gt in gts]
                cer = compute_avg_cer(predictions, gts)
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
                f.write(f"\n======EPOCH {epoch + 1}/{args.epochs}======\n")

            for pad_x, pad_y_shift, pad_y, pad_b, viet_texts, mask in val_dl:
                pad_x = pad_x.to(device)
                pad_y_shift = pad_y_shift.to(device)
                pad_y = pad_y.to(device)
                pad_b = pad_b.to(device)
                mask = mask.to(device)
                logits, l = model(pad_x, viet_texts, pad_y_shift, mask)
                pad_y_shift = pad_y_shift.to("cpu")
                predictions, gts = (
                    greedy_decoding(logits, mask, id_to_token),
                    pad_y_shift_to_string(pad_y_shift, mask, id_to_token),
                )

                predictions = [pred.replace("<unk>", "") for pred in predictions]
                gts = [gt.replace("<unk>", "") for gt in gts]

                if f is not None:
                    for pred, truth in zip(predictions, gts):
                        f.write(f"Prediction: {pred}\tGround Truth: {truth}\n")

                cer = compute_avg_cer(predictions, gts) * len(viet_texts)
                val_cer += cer

            if f is not None:
                f.close()

        avg_val_cer = val_cer / val_size
        scheduler.step(avg_val_cer)

        if best_val_cer > avg_val_cer:
            best_val_cer = avg_val_cer
            best_checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_checkpoint_path)
            logging.info(f"Saved checkpoint to {best_checkpoint_path}")

        logging.info(
            f"Epoch {epoch} Summary:\n"
            f"  Validation CER: {avg_val_cer:.4f}\n"
            f"  Total Validation Samples: {val_size}\n"
            f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        ckpt_path = os.path.join(args.output_dir, "latest.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            ckpt_path,
        )
