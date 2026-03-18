# MediaPipe Face Mesh — Complete Landmark Mapping

MediaPipe Face Mesh provides **478 3D facial landmarks** (468 base + 10 iris with `refine_landmarks=True`).
Each landmark returns `x`, `y`, `z` coordinates:
- `x`, `y`: normalized `[0.0, 1.0]` based on image width/height
- `z`: landmark depth relative to head center (scaled by width in our API)

---

## Quick Reference: Landmark Groups

| Group | Points | Key Indices |
|---|---|---|
| **Mouth Centers** | 4 | `13` (upper inner), `14` (lower inner), `0` (upper outer), `17` (lower outer) |
| **Lips Outer** | 19 | Corners: `61` (left), `291` (right). Top: `17`. Bottom: `0` |
| **Lips Inner** | 19 | Corners: `78` (left), `308` (right). Top: `14`. Bottom: `13` |
| **Left Eye** | 7 | Outer: `33`, Inner: `133`, Iris: `468` |
| **Right Eye** | 7 | Inner: `362`, Outer: `263`, Iris: `473` |
| **Left Eyebrow** | 10 | Inner: `55`/`46`, Arch: `105`, Outer: `70`/`107` |
| **Right Eyebrow** | 10 | Inner: `285`/`276`, Arch: `334`, Outer: `336`/`300` |
| **Nose** | 7 | Tip: `1`, Bridge: `168`/`6`, Nostrils: `129`/`358` |
| **Face Oval** | 36 | Chin: `152`, Jaw contour points |
| **Forehead** | 4 | Center: `10`, Between eyes: `168` |

---

## Detailed Group Breakdown

### Mouth Centers
```
Index  Feature                    Description
─────  ─────────────────────────  ────────────────────────────────
  13   upper_lip_inner_center     Inner midpoint of upper lip
  14   lower_lip_inner_center     Inner midpoint of lower lip
   0   upper_lip_outer_center     Outer midpoint of upper lip
  17   lower_lip_outer_center     Outer midpoint of lower lip
```

### Lips — Outer Contour (19 points, clockwise)
```
Index  Feature          Description
─────  ───────────────  ────────────────────────────────
  61   left_corner      Left mouth corner
 146   upper_left_1     Upper lip, left side
  91   upper_left_2     Upper lip, left side
 181   upper_left_3     Upper lip, left side
  84   upper_left_4     Upper lip, left side
  17   top_center       Top center of outer lip
 314   upper_right_4    Upper lip, right side
 405   upper_right_3    Upper lip, right side
 321   upper_right_2    Upper lip, right side
 375   upper_right_1    Upper lip, right side
 291   right_corner     Right mouth corner
 409   lower_right_1    Lower lip, right side
 270   lower_right_2    Lower lip, right side
 269   lower_right_3    Lower lip, right side
 267   lower_right_4    Lower lip, right side
   0   bottom_center    Bottom center of outer lip
  37   lower_left_4     Lower lip, left side
  39   lower_left_3     Lower lip, left side
  40   lower_left_2     Lower lip, left side
```

### Lips — Inner Contour (19 points, clockwise)
```
Index  Feature          Description
─────  ───────────────  ────────────────────────────────
  78   left_corner      Left inner corner
  95   upper_left_1     Inner upper lip, left
  88   upper_left_2     Inner upper lip, left
 178   upper_left_3     Inner upper lip, left
  87   upper_left_4     Inner upper lip, left
  14   top_center       Inner upper lip center
 317   upper_right_4    Inner upper lip, right
 402   upper_right_3    Inner upper lip, right
 318   upper_right_2    Inner upper lip, right
 324   upper_right_1    Inner upper lip, right
 308   right_corner     Right inner corner
 415   lower_right_1    Inner lower lip, right
 310   lower_right_2    Inner lower lip, right
 311   lower_right_3    Inner lower lip, right
 312   lower_right_4    Inner lower lip, right
  13   bottom_center    Inner lower lip center
  82   lower_left_4     Inner lower lip, left
  81   lower_left_3     Inner lower lip, left
  80   lower_left_2     Inner lower lip, left
```

### Left Eye (First Eye) — 7 points
```
Index  Feature        Description
─────  ─────────────  ────────────────────────────────
  33   outer_corner   Outer corner (toward temple)
 160   upper_lid_1    Upper eyelid point
 158   upper_lid_2    Upper eyelid point
 133   inner_corner   Inner corner (toward nose)
 153   lower_lid_1    Lower eyelid point
 144   lower_lid_2    Lower eyelid point
 468   iris_center    Center of left iris (refined)
```

### Right Eye (Second Eye) — 7 points
```
Index  Feature        Description
─────  ─────────────  ────────────────────────────────
 362   inner_corner   Inner corner (toward nose)
 385   upper_lid_1    Upper eyelid point
 387   upper_lid_2    Upper eyelid point
 263   outer_corner   Outer corner (toward temple)
 373   lower_lid_1    Lower eyelid point
 380   lower_lid_2    Lower eyelid point
 473   iris_center    Center of right iris (refined)
```

### Left Eyebrow — 10 points
```
Index  Feature        Description
─────  ─────────────  ────────────────────────────────
  70   outer_1        Outermost point
  63   outer_2        Outer edge
 105   arch_peak      Highest point of brow arch
  66   inner_1        Inner area
 107   inner_2        Inner edge
  55   inner_edge_1   Innermost (near nose bridge)
  65   inner_edge_2   Lower inner
  52   lower_1        Below brow
  53   lower_2        Below brow
  46   lower_3        Below brow, innermost
```

### Right Eyebrow — 10 points
```
Index  Feature        Description
─────  ─────────────  ────────────────────────────────
 336   outer_1        Outermost point
 296   outer_2        Outer edge
 334   arch_peak      Highest point of brow arch
 293   inner_1        Inner area
 300   inner_2        Inner edge
 285   inner_edge_1   Innermost (near nose bridge)
 295   inner_edge_2   Lower inner
 282   lower_1        Below brow
 283   lower_2        Below brow
 276   lower_3        Below brow, innermost
```

### Nose — 7 points
```
Index  Feature         Description
─────  ──────────────  ────────────────────────────────
   1   tip             Tip of the nose
 168   bridge_top      Top of nose bridge (between brows)
   6   bridge_mid      Middle of nose bridge
 129   left_nostril     Left nostril
 358   right_nostril    Right nostril
  98   left_alar        Left side of nose wing
 327   right_alar       Right side of nose wing
```

---

## API Usage Examples

### Get all landmarks for an image
```python
import requests
r = requests.post("http://localhost:5001/detect/image",
                   files={"image": open("photo.jpg", "rb")})
data = r.json()

# Access specific groups
mouth = data["faces"][0]["landmarks_grouped"]["mouth_centers"]
print(f"Upper lip center: x={mouth['upper_lip_inner_center']['x']}, "
      f"y={mouth['upper_lip_inner_center']['y']}, "
      f"z={mouth['upper_lip_inner_center']['z']}")
```

### Process video frame by frame
```python
r = requests.post("http://localhost:5001/detect/video?sample_every_n=5",
                   files={"image": open("clip.mp4", "rb")})
data = r.json()

for frame in data["frames"]:
    for face in frame["faces"]:
        left_eye = face["landmarks_grouped"]["left_eye"]
        print(f"Frame {frame['frame_number']}: "
              f"iris at ({left_eye['iris_center']['x']}, {left_eye['iris_center']['y']})")
```

### Get the full landmark map
```python
r = requests.get("http://localhost:5001/landmarks/map")
print(r.json())
```
