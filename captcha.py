import os
import random
import math
from PIL import Image, ImageDraw, ImageFont

def generate_exact_match_captcha(bg_path=r"D:\meroshare\assets\blankCaptcha.png",
                                 output_dir=r"D:\meroshare\assets\synthetic",
                                 num_images=5):
    os.makedirs(output_dir, exist_ok=True)
    
    font_path = r"C:\Windows\Fonts\timesbd.ttf"
    try:
        # 34-36 is the perfect scale for a 40px tall canvas
        font = ImageFont.truetype(font_path, 35)  
    except IOError:
        print(f"Bold serif font not found at {font_path}. Falling back to default.")
        font = ImageFont.load_default()

    for _ in range(num_images):
        # 1. Load background grid and convert to RGBA to support alpha blending layers
        captcha_img = Image.open(bg_path).convert("RGBA")
        img_w, img_h = captcha_img.size
        
        # 2. Create a separate, transparent overlay for the line/smudge effects
        effect_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw_effect = ImageDraw.Draw(effect_layer)
        
        # Create a direct draw interface for the text base layer
        draw_base = ImageDraw.Draw(captcha_img)
        
        # Generate a random 5-digit sequence
        label = "".join([str(random.randint(0, 9)) for _ in range(5)])
        
        # 3. Draw Text (Slightly relaxed character spacing to prevent numbers flipping order)
        x_pos = 14  
        for char in label:
            y_pos = random.randint(1, 4)  # Kept balanced to fit the 40px container
            draw_base.text((x_pos, y_pos), char, fill="black", font=font)
            # 21-23px advance step maintains natural overlapping and fusing at font size 35
            x_pos += random.randint(21, 23) 
            
        # 4. Draw the thin tracking ellipse directly on the base layer
        ellipse_x1 = random.randint(25, 45)
        ellipse_y1 = random.randint(2, 6)
        ellipse_x2 = random.randint(120, 135)
        ellipse_y2 = random.randint(34, 38)
        draw_base.ellipse([ellipse_x1, ellipse_y1, ellipse_x2, ellipse_y2], outline="black", width=1)
        
        # 5. Construct the Wavy Strike-Through Line using transparency
        line_points = []
        wave_frequency = random.uniform(0.04, 0.05)
        wave_amplitude = random.uniform(4.5, 5.5)
        y_center = img_h // 2 + random.randint(-2, 1)
        
        for x in range(0, img_w):
            y = y_center + int(wave_amplitude * math.sin(wave_frequency * x))
            line_points.append((x, y))
            
        # Draw line with moderate width and medium opacity (Alpha 140/255)
        # This prevents it from blooming into a massive solid block during thresholding
        draw_effect.line(line_points, fill=(0, 0, 0, 140), width=4)
        
        # 6. Inject the central organic ink bleed/smudge using soft opacity gradients
        blob_x = img_w // 2 + random.randint(-12, 12)
        blob_y = y_center + random.randint(-1, 1)
        
        # Draw a semi-translucent inner core smudge
        draw_effect.ellipse([blob_x - 14, blob_y - 4, blob_x + 14, blob_y + 4], fill=(0, 0, 0, 150))
        # Draw a softer, outer bleeding boundary halo
        draw_effect.ellipse([blob_x - 22, blob_y - 6, blob_x + 22, blob_y + 6], fill=(0, 0, 0, 85))
        
        # 7. Alpha-composite the effect overlay back onto the text/grid canvas
        final_composite = Image.alpha_composite(captcha_img, effect_layer)
        
        # 8. Convert back to Grayscale and apply clean Binarization Threshold
        final_gray = final_composite.convert("L")
        
        # A threshold of 110-125 guarantees the background lines stay intact, 
        # while transforming soft transparent edges into realistic text-bleeding marks
        binary_img = final_gray.point(lambda p: 255 if p > 115 else 0, mode='1')
        
        # Save output named exactly as its target label
        save_path = os.path.join(output_dir, f"{label}.png")
        binary_img.save(save_path)

    print(f"Generated {num_images} exact-match CAPTCHAs in '{output_dir}'.")