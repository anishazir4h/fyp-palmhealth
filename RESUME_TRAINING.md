# Resume Training Guide

## How Checkpoint Resuming Works

The updated `train_ganoderma.py` now automatically detects if a checkpoint exists and asks if you want to resume.

### Automatic Resume

When you run training:

```bash
python train_ganoderma.py
```

If a checkpoint exists at `runs/detect/Ganoderma_FasterRCNN/weights/last.pt`, you'll see:

```
📂 Found existing checkpoint: runs/detect/Ganoderma_FasterRCNN/weights/last.pt
   Resume from last checkpoint? (y/n):
```

- Type **`y`** to continue from where you left off
- Type **`n`** to start fresh training (overwrites checkpoints)

### What Gets Saved

Each checkpoint includes:
- ✅ Model weights
- ✅ Optimizer state (momentum, learning rate, etc.)
- ✅ Learning rate scheduler state
- ✅ Current epoch number
- ✅ Best validation loss so far
- ✅ Training and validation losses

### Files Saved

- **`best.pt`** - Best model based on validation loss (with full training state)
- **`last.pt`** - Most recent epoch (with full training state)

### Resume Example

```
Original training: Epochs 0 → 50
   Stopped at epoch 25 (last.pt saved)

Resume training:
   📂 Found existing checkpoint
   Resume from last checkpoint? y
   ✅ Resumed from epoch 26
   Best loss so far: 0.3542

   Continues: Epochs 26 → 50
```

### Benefits

✅ **No wasted progress** - Continue exactly where you left off
✅ **Optimizer state preserved** - Learning continues smoothly
✅ **Best model tracked** - Always keeps the best performing model
✅ **Flexible** - Can choose to resume or start fresh

## Quick Commands

**Start new training:**
```bash
python train_ganoderma.py
# When prompted: n (or delete checkpoint folder first)
```

**Continue training:**
```bash
python train_ganoderma.py
# When prompted: y
```

**Check if checkpoint exists:**
```bash
ls runs/detect/Ganoderma_FasterRCNN/weights/
```
