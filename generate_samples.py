from shape_generator import ShapeGenerator
import sys

if __name__ == "__main__":
    num_samples = 5
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    
    print(f"Generating {num_samples} sample images for each shape...")
    ShapeGenerator.save_sample_images(samples=num_samples)
    print(f"Done! Created {num_samples * 8} sample images in sample_images/shapes/")
