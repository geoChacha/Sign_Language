"""
Extract keypoints for all 100 signs in NSLT-100 dataset
Run this first, then generate animations separately if needed
"""
from extract_keypoints import extract_nslt_100

if __name__ == '__main__':
    print("\n Starting FULL extraction for all 100 signs...")
    print("This will extract keypoints from all videos in nslt_100.json\n")
    
    results, summary = extract_nslt_100(
        output_dir='keypoints_full',
        max_signs=None  # Extract ALL signs (no limit)
    )
    
    print("\n Full extraction complete!")
    print("\nNext steps:")
    print("1. Check keypoints_full/signs/ for all extracted files")
    print("2. Review keypoints_full/extraction_results.json for quality metrics")
    print("3. Generate VRM animations:")
    print("   python generate_vrm_animations.py")
    print("\n r run the full pipeline:")
    print("   python run_full_extraction.py")
