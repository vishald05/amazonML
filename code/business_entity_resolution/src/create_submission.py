"""Create final submission ZIP file."""

import zipfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import PROJECT_ROOT, OUTPUT_DIR, CODE_DIR

def create_submission_zip(team_name="team"):
    """Create submission ZIP file with required structure."""

    zip_filename = PROJECT_ROOT / f"{team_name}_submission.zip"

    print(f"Creating submission ZIP: {zip_filename}")
    print("=" * 60)

    # Files to include
    files_to_include = {
        # Output files (required)
        'output/matching_results.tsv': OUTPUT_DIR / 'matching_results.tsv',
        'output/candidate_pairs.tsv': OUTPUT_DIR / 'candidate_pairs.tsv',

        # Code (required)
        'code/business_entity_resolution/src/config.py': CODE_DIR / 'src' / 'config.py',
        'code/business_entity_resolution/src/io_utils.py': CODE_DIR / 'src' / 'io_utils.py',
        'code/business_entity_resolution/src/normalization.py': CODE_DIR / 'src' / 'normalization.py',
        'code/business_entity_resolution/src/features.py': CODE_DIR / 'src' / 'features.py',
        'code/business_entity_resolution/src/blocking.py': CODE_DIR / 'src' / 'blocking.py',
        'code/business_entity_resolution/src/pair_generation.py': CODE_DIR / 'src' / 'pair_generation.py',
        'code/business_entity_resolution/src/evaluate.py': CODE_DIR / 'src' / 'evaluate.py',
        'code/business_entity_resolution/src/train.py': CODE_DIR / 'src' / 'train.py',
        'code/business_entity_resolution/src/inference.py': CODE_DIR / 'src' / 'inference.py',
        'code/business_entity_resolution/src/sqlite_blocking.py': CODE_DIR / 'src' / 'sqlite_blocking.py',
        'code/business_entity_resolution/src/scalable_pipeline.py': CODE_DIR / 'src' / 'scalable_pipeline.py',
        'code/business_entity_resolution/src/eda.py': CODE_DIR / 'src' / 'eda.py',
        'code/business_entity_resolution/README.md': CODE_DIR / 'README.md',
        'code/business_entity_resolution/requirements.txt': CODE_DIR / 'requirements.txt',

        # Documentation (required)
        'Documentation_template.md': PROJECT_ROOT / 'Documentation_template.md',
    }

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for zip_path, file_path in files_to_include.items():
            if file_path.exists():
                print(f"  Adding: {zip_path}")
                zipf.write(file_path, zip_path)
            else:
                print(f"  WARNING: Missing {file_path}")

    print("=" * 60)
    print(f"Submission ZIP created: {zip_filename}")
    print(f"Size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
    print("\nNext steps:")
    print("1. Update Documentation_template.md with final results")
    print("2. Validate the ZIP structure")
    print("3. Submit!")

    return zip_filename


if __name__ == "__main__":
    import sys
    team_name = sys.argv[1] if len(sys.argv) > 1 else "team"
    create_submission_zip(team_name)
