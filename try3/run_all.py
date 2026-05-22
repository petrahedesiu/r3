
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))


def main():
    start = time.time()

    print("\n" + "#" * 70)
    print("# TRY3 FULL PIPELINE")
    print("#" * 70)

    print("\n\n>>> STAGE 1: Training coarse localizer...")
    from train_coarse import main as train_coarse
    coarseResults = train_coarse()

    print("\n\n>>> STAGE 2a: Training AEAL specialist...")
    from train_aeal import main as train_aeal
    aealResults = train_aeal()

    print("\n\n>>> STAGE 2b: Training AEAR specialist...")
    from train_aear import main as train_aear
    aearResults = train_aear()

    print("\n\n>>> ENSEMBLE EVALUATION...")
    from eval_ensemble import main as eval_ensemble
    ensembleResults = eval_ensemble()

    elapsed = time.time() - start
    print("\n\n" + "#" * 70)
    print(f"# TRY3 COMPLETE ({elapsed / 60:.1f} minutes)")
    print("#" * 70)

    print(f"\nCoarse  best dice: {coarseResults['best_val_dice']:.4f}")
    print(f"AEAL    best dice: {aealResults['best_val_dice']:.4f}")
    print(f"AEAR    best dice: {aearResults['best_val_dice']:.4f}")

    overall = ensembleResults.get("overall_fg_slices", ensembleResults.get("overall_all_slices", {}))
    print(f"\nEnsemble FG Dice:      {overall.get('mean_fg_dice', 'N/A')}")
    print(f"Ensemble FG Recall:    {overall.get('mean_fg_recall', 'N/A')}")
    print(f"Ensemble AEAL Dice:    {overall.get('dice_per_class', {}).get('1', 'N/A')}")
    print(f"Ensemble AEAR Dice:    {overall.get('dice_per_class', {}).get('2', 'N/A')}")


if __name__ == "__main__":
    main()
